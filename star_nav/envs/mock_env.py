"""Dependency-free synthetic stand-in for the AirSim/Unreal Engine oil-palm
corridor simulator described in Section 4.

This is NOT a photorealistic renderer. It exists so the full STAR-Nav
pipeline (SACR -> CAMR -> AGSS-PPO -> training -> evaluation) is runnable
end-to-end on a laptop with no Unreal Engine, AirSim, or GPU rendering
stack. It reproduces the paper's *task structure* faithfully:

  * procedurally placed palm rows, with the same tree/row spacing mean
    and std used in Table 6 (Scenario A/B/C -> low/medium/high aliasing
    via decreasing spacing variance),
  * monocular forward-facing perception via per-column ray casting
    (distance to nearest trunk), producing an RGB-like image, a
    segmentation mask, and a dense depth map from a single shared
    geometric model (so SACR's segmentation/geometry/depth branches are
    all learning consistent, correlated signals, as they would from a
    real camera),
  * the same POMDP action space (v_x, v_y, v_z, omega), reward
    decomposition (progress / smoothness / alive), and termination
    conditions (goal reached / collision / timeout) as Section 3.4.

Swap in `AirSimCorridorEnv` for the real simulator once Unreal Engine +
AirSim + the custom plantation assets are available; both share the
`BaseCorridorEnv` interface, so no other code needs to change.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base_env import BaseCorridorEnv, EnvObservation, EnvStepResult, PrivilegedInfo

# Segmentation class ids (kept tiny; matches sacr.num_seg_classes=5 default)
CLASS_SKY, CLASS_GROUND, CLASS_TRUNK, CLASS_CANOPY, CLASS_ACTOR = range(5)

# Dynamic "worker" actors (moving people, CLASS_ACTOR) -- mirrors the moving
# workers in the Gazebo capture env (ros_gazebo_bridge/perception_capture) so
# the PPO policy + AGSS shield learn to avoid moving PEOPLE, not just static
# trunks. Actors are extra circular obstacles in the ray-cast: they show up in
# depth / seg (CLASS_ACTOR) / rgb, count toward collision, and walk across the
# lane each step.
N_ACTORS = 5
ACTOR_RADIUS = 0.35
ACTOR_SPEED = 0.9          # m/s lateral walking speed
ACTOR_LANE_HALF = 3.0      # actors wander within +/- this of the corridor centre (m)

# Reward shaping for collision-avoidant navigation (Section 3.4 reward). The
# original sparse -5 collision penalty was too weak vs accumulated forward
# progress, so PPO learned to sprint-and-crash; a stronger terminal penalty +
# a dense proximity penalty fixes the incentive.
COLLISION_PENALTY = 30.0
SUCCESS_BONUS = 15.0       # terminal reward for reaching the goal (pulls the policy forward)
CLEAR_MARGIN = 1.0         # m; proximity penalty ramps up below this clearance
CLEAR_COEF = 0.15          # gentle -- a nudge to keep distance, not to dominate progress

# Per-mission lateral boundaries: a virtual lane of half-width `lane_half` about
# the corridor centre. Crossing it ends the episode with BOUNDARY_PENALTY, so the
# policy learns to stay on the intended path instead of exploring the whole map
# (a big speed-up, especially early in a curriculum). Default (no boundary set)
# uses DEFAULT_LANE_HALF just as the old off-lane sanity threshold.
BOUNDARY_PENALTY = 10.0
DEFAULT_LANE_HALF = 4.0

# Anticipatory occupancy ground truth (CAMR novelty). For each step, look
# `horizon` steps into the recorded future and mark whether any dynamic actor
# will fall within `lookahead` metres ahead of the drone and inside +/- `lane`
# metres laterally, split into the drone's left (body +y) and right (body -y).
OCC_HORIZON = 8
OCC_LOOKAHEAD = 6.0
OCC_LANE = 2.5


def compute_future_occupancy(drone_xy, drone_yaw, actor_xy_seq,
                             horizon=OCC_HORIZON, lookahead=OCC_LOOKAHEAD, lane=OCC_LANE):
    """Future actor-occupancy target (T, 2) = [left, right] in {0,1}.

    For step t, transform every actor position at steps t+1..t+horizon into the
    drone's body frame at t (forward = +x, left = +y) and flag left/right if an
    actor lies ahead within `lookahead` m and |lateral| < `lane` m. This is the
    supervision for CAMR.predict_occupancy -- the belief learns to anticipate
    where a moving worker WILL be. Backend-agnostic: works from any recorded
    (drone pose, actor poses) trajectory (Mock now, Gazebo actor poses later).

    Args:
        drone_xy: (T, 2) drone position per step.
        drone_yaw: (T,) drone heading per step (rad).
        actor_xy_seq: length-T list of (n_t, 2) actor positions (n_t may be 0).
    """
    drone_xy = np.asarray(drone_xy, dtype=np.float32)
    drone_yaw = np.asarray(drone_yaw, dtype=np.float32)
    T = len(drone_xy)
    occ = np.zeros((T, 2), dtype=np.float32)
    for t in range(T):
        c, s = np.cos(drone_yaw[t]), np.sin(drone_yaw[t])
        for k in range(1, horizon + 1):
            tk = t + k
            if tk >= T:
                break
            a = actor_xy_seq[tk]
            if a is None or len(a) == 0:
                continue
            rel = np.asarray(a, dtype=np.float32) - drone_xy[t]
            fwd = c * rel[:, 0] + s * rel[:, 1]        # body +x (forward)
            lat = -s * rel[:, 0] + c * rel[:, 1]       # body +y (left)
            inview = (fwd > 0) & (fwd < lookahead) & (np.abs(lat) < lane)
            if np.any(inview & (lat > 0)):
                occ[t, 0] = 1.0                        # left
            if np.any(inview & (lat < 0)):
                occ[t, 1] = 1.0                        # right
    return occ

WEATHER_PARAMS = {
    # (brightness_mult, noise_std, canopy_contrast_mult)
    "clear_day": (1.00, 0.02, 1.00),
    "cloudy_day": (0.85, 0.05, 0.85),
    "clear_afternoon": (0.90, 0.03, 0.90),
    "cloudy_afternoon": (0.75, 0.07, 0.70),
}


@dataclass
class _World:
    tree_xy: np.ndarray       # (N, 2) trunk centers in metres
    trunk_radius: float
    corridor_center_y: float
    goal_xy: np.ndarray


class MockCorridorEnv(BaseCorridorEnv):
    def __init__(self, cfg):
        self.cfg = cfg
        self.area = cfg.area_size_m
        self.max_steps = cfg.episode_max_steps
        # Opt-in long winding ROUTE (star_nav/envs/routes.py). Without it every
        # behaviour below is byte-identical to the original straight/zigzag world.
        # A route also makes the map a long STRIP, so x and y extents differ.
        self._route = None
        self._area_x = self._area_y = float(self.area)
        _route_name = getattr(cfg, "route", None)
        if _route_name:
            from .routes import make_route
            self._route = make_route(_route_name)
            self._area_x = self._route.length_x
            self._area_y = self._route.width_y
        self.img_w, self.img_h = 160, 120  # internal render resolution (kept small for CPU speed)
        self.n_rays = self.img_w
        self.fov_deg = 90.0
        self.max_range = 15.0
        self.dt = 0.2  # seconds per control step

        self._max_forward_speed = 2.5
        self._max_yaw_rate = np.deg2rad(120.0)

        # Zigzag centerline: a PIECEWISE path -- straight, jog right, straight,
        # jog left, straight, jog right, straight -- so the corridor doglegs down
        # the plantation (not a smooth sine). `zigzag_amp` is the lateral jog
        # amplitude in metres (0 -> a straight corridor, byte-identical to the
        # original). The tree rows follow the doglegs, phi (theta_corr_gt[0]) is
        # the heading error vs the LOCAL segment tangent, and lane deviation is
        # the perpendicular distance to the path -- so perception must read the
        # turning corridor and the policy must steer through the turns. Moving
        # workers (CLASS_ACTOR) are scattered along it (set env.n_actors=5-6).
        self._zz_amp = float(getattr(cfg, "zigzag_amp", 0.0))
        # Along-x fractions of the anchor points and their lateral offset (in
        # units of _zz_amp): 0,0 (straight) -> -1 (right) -> -1 (straight) ->
        # +1 (left) -> +1 (straight) -> 0 (right back) -> 0 (straight to goal).
        # A LONG initial straight (first anchor gap ~1/3 of the corridor) gives the
        # curriculum an on-ramp: the early short-goal stages are pure straight
        # (easy, build confidence), and the first dogleg only appears once the goal
        # extends past it -- mirroring how the straight-corridor curriculum learned.
        # Turns are spread over wide x-gaps (esp. the middle 2*amp cross-over) so
        # the doglegs stay gentle (~40 deg at amp=5) rather than sharp.
        self._zz_fx = np.array([0.00, 0.32, 0.44, 0.52, 0.72, 0.80, 0.92, 1.00], dtype=np.float32)
        self._zz_off = np.array([0.0, 0.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self._base_center_y = 0.0
        self._zz_x = None   # anchor x positions (set per world in _generate_world)
        self._zz_y = None   # anchor y positions

        self.rng = np.random.default_rng()
        self.n_actors = int(getattr(cfg, "n_actors", N_ACTORS))
        self._goal_x = None          # curriculum: None -> full corridor (area-2)
        self._lane_half = None       # per-mission lateral boundary; None -> DEFAULT_LANE_HALF
        self._world: _World | None = None
        self._actor_xy = None       # (A, 2) moving-people positions
        self._actor_vel = None      # (A, 2) velocities
        self._pos = np.zeros(2)
        self._heading = 0.0
        self._t = 0
        self._prev_omega = 0.0
        self._prev_v = np.zeros(3)
        self._weather = "clear_day"

    @property
    def max_forward_speed(self) -> float:
        return self._max_forward_speed

    # ------------------------------------------------------------------
    # Zigzag centerline helpers (straight corridor when _zz_amp == 0)
    # ------------------------------------------------------------------
    def _center_y(self, x):
        """Corridor centerline y at along-corridor position x (scalar or array),
        piecewise-linear through the zigzag anchors. Straight when _zz_x is None."""
        if self._route is not None:
            return self._route.center_y(x)
        if self._zz_x is None:
            return self._base_center_y + 0.0 * np.asarray(x, dtype=np.float32)
        return np.interp(np.asarray(x, dtype=np.float32), self._zz_x, self._zz_y).astype(np.float32)

    def _tangent(self, x):
        """Local segment tangent angle (rad) at x -- the 'corridor forward' of the
        dogleg the drone is currently on. 0 for a straight corridor."""
        if self._route is not None:
            return self._route.tangent(x)
        if self._zz_x is None:
            return np.zeros_like(np.asarray(x, dtype=np.float32)) if np.ndim(x) else 0.0
        xa = np.asarray(x, dtype=np.float32)
        i = np.clip(np.searchsorted(self._zz_x, xa, side="right") - 1, 0, len(self._zz_x) - 2)
        dx = self._zz_x[i + 1] - self._zz_x[i]
        dy = self._zz_y[i + 1] - self._zz_y[i]
        ang = np.arctan2(dy, np.maximum(dx, 1e-6))
        return float(ang) if np.ndim(x) == 0 else ang.astype(np.float32)

    # ------------------------------------------------------------------
    # World generation (Table 6 parameters)
    # ------------------------------------------------------------------
    def _generate_world(self, scenario: str) -> _World:
        sc = self.cfg.scenarios[scenario]
        n_rows = max(2, int(self._area_y / sc.row_spacing_mean))
        trees = []
        for r in range(n_rows):
            row_y = r * sc.row_spacing_mean + self.rng.normal(0, sc.row_spacing_std)
            n_trees = max(2, int(self._area_x / sc.tree_spacing_mean))
            x = 0.0
            for _ in range(n_trees):
                x += max(1.0, self.rng.normal(sc.tree_spacing_mean, sc.tree_spacing_std))
                trees.append((x, row_y))
        tree_xy = np.array(trees, dtype=np.float32)
        corridor_center_y = (n_rows * sc.row_spacing_mean) / 2.0
        if self._route is not None:
            corridor_center_y = float(self._route.center_y(0.0))   # route sets its own centre
        self._base_center_y = corridor_center_y
        # Build the piecewise zigzag anchors for this world (None -> straight).
        if self._zz_amp > 0.0:
            self._zz_x = (self._zz_fx * self.area).astype(np.float32)
            self._zz_y = (corridor_center_y + self._zz_off * self._zz_amp).astype(np.float32)
        else:
            self._zz_x = self._zz_y = None
        # Bend the whole plantation so the clear corridor down the middle follows
        # the doglegs: shift each tree's y by the centerline offset at its own x
        # (rows keep their spacing). No-op when straight.
        tree_xy[:, 1] = tree_xy[:, 1] + (self._center_y(tree_xy[:, 0]) - corridor_center_y)
        goal_x = self._goal_x if self._goal_x is not None else self._area_x - 2.0
        goal_xy = np.array([goal_x, float(self._center_y(goal_x))], dtype=np.float32)
        return _World(tree_xy=tree_xy, trunk_radius=0.25, corridor_center_y=corridor_center_y, goal_xy=goal_xy)

    def set_curriculum(self, goal_x=None, n_actors=None, lane_half=None) -> None:
        """Adjust mission difficulty for curriculum learning (takes effect on the
        next reset): goal distance, number of moving people, and lateral lane
        boundary half-width."""
        if goal_x is not None:
            self._goal_x = float(goal_x)
        if n_actors is not None:
            self.n_actors = int(n_actors)
        if lane_half is not None:
            self._lane_half = float(lane_half)

    # ------------------------------------------------------------------
    # Dynamic actors (moving people, CLASS_ACTOR) -- mirrors the Gazebo workers
    # ------------------------------------------------------------------
    def _init_actors(self) -> None:
        w = self._world
        if self.n_actors <= 0:
            self._actor_xy = np.zeros((0, 2), dtype=np.float32)
            self._actor_vel = np.zeros((0, 2), dtype=np.float32)
            return
        # scatter along the corridor ahead of the drone, inside the lane
        xs = np.linspace(6.0, self._area_x - 6.0, self.n_actors) + self.rng.normal(0, 1.0, self.n_actors)
        ys = self._center_y(xs) + self.rng.uniform(-ACTOR_LANE_HALF, ACTOR_LANE_HALF, self.n_actors)
        self._actor_xy = np.stack([xs, ys], axis=1).astype(np.float32)
        # mostly-lateral walking (crossing the lane) + a little along-corridor drift
        ang = self.rng.uniform(0, 2 * np.pi, self.n_actors)
        self._actor_vel = np.stack(
            [0.3 * ACTOR_SPEED * np.cos(ang), ACTOR_SPEED * np.sin(ang)], axis=1).astype(np.float32)

    def _update_actors(self) -> None:
        if self._actor_xy is None or len(self._actor_xy) == 0:
            return
        w = self._world
        self._actor_xy = self._actor_xy + self._actor_vel * self.dt
        # Bounce laterally within +-ACTOR_LANE_HALF about the (possibly bent)
        # centerline at each actor's own x, so workers cross the curving corridor.
        cy = self._center_y(self._actor_xy[:, 0])
        lo, hi = cy - ACTOR_LANE_HALF, cy + ACTOR_LANE_HALF
        bounce = (self._actor_xy[:, 1] > hi) | (self._actor_xy[:, 1] < lo)
        self._actor_vel[bounce, 1] *= -1.0
        self._actor_xy[:, 1] = np.clip(self._actor_xy[:, 1], lo, hi)
        self._actor_xy[:, 0] = np.clip(self._actor_xy[:, 0], 3.0, self._area_x - 3.0)

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(self, scenario: str = "A", weather: str = "clear_day") -> EnvObservation:
        self._scenario = scenario
        self._weather = weather
        self._world = self._generate_world(scenario)
        self._pos = np.array([2.0, float(self._center_y(2.0)) + self.rng.normal(0, 0.3)], dtype=np.float32)
        self._heading = float(self._tangent(2.0))
        self._t = 0
        self._prev_omega = 0.0
        self._prev_v = np.zeros(3, dtype=np.float32)
        self._init_actors()
        self._prev_goal_dist = self._goal_distance()
        obs, _ = self._render_and_info()
        return obs

    def step(self, action: np.ndarray) -> EnvStepResult:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        v_x, v_y, v_z, omega_n = action
        v_x = v_x * self._max_forward_speed
        v_y = v_y * self._max_forward_speed
        omega = omega_n * self._max_yaw_rate

        # Simple unicycle-ish kinematic integration in the horizontal plane.
        self._heading += omega * self.dt
        dx = (v_x * np.cos(self._heading) - v_y * np.sin(self._heading)) * self.dt
        dy = (v_x * np.sin(self._heading) + v_y * np.cos(self._heading)) * self.dt
        self._pos = self._pos + np.array([dx, dy], dtype=np.float32)
        self._t += 1
        self._update_actors()

        obs, info = self._render_and_info()

        goal_dist = info.goal_distance
        r_progress = (self._prev_goal_dist - goal_dist) / self._max_forward_speed
        d_omega = omega - self._prev_omega
        d_v = np.array([v_x, v_y, v_z], dtype=np.float32) - self._prev_v
        r_smooth = -abs(d_omega) - float(np.linalg.norm(d_v))
        r_alive = 0.01
        # Dense proximity penalty: ramps up as the nearest obstacle (trunk OR
        # moving person) comes within CLEAR_MARGIN. Without this, the only
        # obstacle signal is the sparse terminal collision penalty, and PPO
        # learns to sprint-and-crash (progress reward > rare crash penalty).
        clearance = float(info.depth.min())
        r_clear = -CLEAR_COEF * max(0.0, CLEAR_MARGIN - clearance)
        reward = 1.0 * r_progress + 0.05 * r_smooth + 0.01 * r_alive + r_clear

        self._prev_goal_dist = goal_dist
        self._prev_omega = omega
        self._prev_v = np.array([v_x, v_y, v_z], dtype=np.float32)

        success = goal_dist < 1.5
        timeout = self._t >= self.max_steps
        # Leaving the lane boundary ends the mission with a penalty, so the
        # policy learns to stay on the intended path (focused training).
        done = bool(success or info.collided or timeout or info.off_lane)
        if info.collided:
            reward -= COLLISION_PENALTY
        if info.off_lane:
            reward -= BOUNDARY_PENALTY
        if success:
            reward += SUCCESS_BONUS

        return EnvStepResult(obs=obs, reward=float(reward), done=done, success=success, info=info)

    # ------------------------------------------------------------------
    # Rendering: per-column ray casting against trunk circles
    # ------------------------------------------------------------------
    def _goal_distance(self) -> float:
        if self._route is None:
            return float(np.linalg.norm(self._pos - self._world.goal_xy))
        # On a meandering route Euclidean distance is a BROKEN progress signal: you
        # can be metres from the goal in a straight line with hundreds of metres of
        # corridor left, and a meander can even carry you *away* from the goal --
        # which would make the progress reward punish flying the corridor correctly.
        # Measure what the drone actually has to fly: remaining ARC LENGTH (+ how
        # far off the centreline it is, so the goal is only "reached" on the path).
        gx = self._goal_x if self._goal_x is not None else (self._area_x - 2.0)
        s_now = float(self._route.arclen(np.clip(self._pos[0], 0.0, self._area_x)))
        s_goal = float(self._route.arclen(gx))
        lat = abs(float(self._pos[1] - self._center_y(self._pos[0])))
        return float(max(0.0, s_goal - s_now) + lat)

    def _render_and_info(self) -> tuple[EnvObservation, PrivilegedInfo]:
        w = self._world
        half_fov = np.deg2rad(self.fov_deg) / 2.0
        angles = self._heading + np.linspace(-half_fov, half_fov, self.n_rays)

        depths = np.full(self.n_rays, self.max_range, dtype=np.float32)
        hit_trunk = np.zeros(self.n_rays, dtype=bool)

        rel = w.tree_xy - self._pos[None, :]
        dist_to_center = np.linalg.norm(rel, axis=1)
        near_mask = dist_to_center < (self.max_range + w.trunk_radius)
        candidates = rel[near_mask]

        if len(candidates) > 0:
            cand_angle = np.arctan2(candidates[:, 1], candidates[:, 0])
            cand_dist = np.linalg.norm(candidates, axis=1)
            for i, ray_angle in enumerate(angles):
                d_angle = np.abs(np.arctan2(np.sin(cand_angle - ray_angle), np.cos(cand_angle - ray_angle)))
                angular_radius = np.clip(w.trunk_radius / np.maximum(cand_dist, 1e-3), 0, np.pi / 2)
                hit = d_angle < angular_radius
                if np.any(hit):
                    depths[i] = cand_dist[hit].min()
                    hit_trunk[i] = True

        # Moving people: same circular ray-cast; an actor occludes a trunk on
        # rays where it is the nearer hit (so it segments as CLASS_ACTOR there).
        hit_actor = np.zeros(self.n_rays, dtype=bool)
        if self._actor_xy is not None and len(self._actor_xy) > 0:
            arel = self._actor_xy - self._pos[None, :]
            adist = np.linalg.norm(arel, axis=1)
            acand = arel[adist < (self.max_range + ACTOR_RADIUS)]
            acdist = adist[adist < (self.max_range + ACTOR_RADIUS)]
            if len(acand) > 0:
                acang = np.arctan2(acand[:, 1], acand[:, 0])
                for i, ray_angle in enumerate(angles):
                    da = np.abs(np.arctan2(np.sin(acang - ray_angle), np.cos(acang - ray_angle)))
                    arad = np.clip(ACTOR_RADIUS / np.maximum(acdist, 1e-3), 0, np.pi / 2)
                    hit = da < arad
                    if np.any(hit):
                        d = float(acdist[hit].min())
                        if d < depths[i]:
                            depths[i] = d
                            hit_actor[i] = True
                            hit_trunk[i] = False

        brightness, noise_std, contrast = WEATHER_PARAMS[self._weather]

        # Depth map: broadcast the 1D ray-cast profile into a (H, W) image
        # (constant along image rows -- a flat-ground simplification).
        depth_map = np.tile(depths[None, :], (self.img_h, 1))
        depth_map += self.rng.normal(0, noise_std * 2.0, depth_map.shape).astype(np.float32)
        depth_map = np.clip(depth_map, 0.05, self.max_range)

        seg_mask = np.full((self.img_h, self.img_w), CLASS_GROUND, dtype=np.int64)
        horizon = int(self.img_h * 0.45)
        seg_mask[:horizon, :] = CLASS_SKY
        trunk_cols = np.where(hit_trunk)[0]
        for c in trunk_cols:
            top = int(horizon - (1.0 - depths[c] / self.max_range) * horizon * 0.9)
            seg_mask[max(0, top):horizon, c] = CLASS_CANOPY
            seg_mask[horizon:, c] = np.where(depths[c] < 2.0, CLASS_TRUNK, CLASS_GROUND)

        # standing people occupy a vertical strip from the ground up, taller
        # (more image rows) the nearer they are.
        actor_strips = []
        for c in np.where(hit_actor)[0]:
            person_h = int((self.img_h - horizon) * float(np.clip(3.0 / max(depths[c], 0.5), 0.4, 2.6)))
            top = max(0, self.img_h - person_h)
            seg_mask[top:, c] = CLASS_ACTOR
            actor_strips.append((c, top))

        norm_depth = 1.0 - (depth_map / self.max_range)
        rgb = np.zeros((self.img_h, self.img_w, 3), dtype=np.float32)
        rgb[..., 1] = 0.3 + 0.5 * norm_depth * contrast          # greener where closer to canopy
        rgb[..., 0] = 0.25 + 0.3 * (1 - norm_depth)              # brown ground/trunk tint
        rgb[:horizon, :, 2] = 0.6                                 # sky tint
        rgb *= brightness
        rgb += self.rng.normal(0, noise_std, rgb.shape).astype(np.float32)
        for c, top in actor_strips:                      # hi-vis orange worker
            rgb[top:, c, 0] = 0.9 * brightness
            rgb[top:, c, 1] = 0.45 * brightness
            rgb[top:, c, 2] = 0.10 * brightness
        rgb = np.clip(rgb, 0, 1)
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        d_left = float(depths[: self.n_rays // 3].mean())
        d_center = float(depths[self.n_rays // 3: 2 * self.n_rays // 3].mean())
        d_right = float(depths[2 * self.n_rays // 3:].mean())
        d_forward = float(depths[self.n_rays // 2])

        # Heading error and lane deviation relative to the LOCAL centerline (bent
        # for zigzag; reduces to the straight-corridor case when _zz_amp == 0).
        c_tan = float(self._tangent(self._pos[0]))
        phi = float(np.arctan2(np.sin(self._heading - c_tan), np.cos(self._heading - c_tan)))
        lateral_dev = float((self._pos[1] - self._center_y(self._pos[0])) * np.cos(c_tan))
        collided = bool(np.any(dist_to_center < w.trunk_radius + 0.15))
        if self._actor_xy is not None and len(self._actor_xy) > 0:      # hitting a person counts too
            adist_body = np.linalg.norm(self._actor_xy - self._pos[None, :], axis=1)
            collided = collided or bool(np.any(adist_body < ACTOR_RADIUS + 0.15))
        lane_half = self._lane_half if self._lane_half is not None else DEFAULT_LANE_HALF
        off_lane = bool(abs(lateral_dev) > lane_half)

        pose = np.array([self._pos[0], self._pos[1], 0.0,
                          0.0, 0.0, np.sin(self._heading / 2), np.cos(self._heading / 2)], dtype=np.float32)
        imu = np.concatenate([
            np.array([0.0, 0.0, 9.81], dtype=np.float32) + self.rng.normal(0, 0.05, 3).astype(np.float32),
            np.array([0.0, 0.0, self._prev_omega], dtype=np.float32) + self.rng.normal(0, 0.02, 3).astype(np.float32),
        ])

        obs = EnvObservation(rgb=rgb_uint8, pose=pose, imu=imu)
        info = PrivilegedInfo(
            seg_mask=seg_mask,
            depth=depth_map,
            theta_corr_gt=np.array([phi, d_left, d_right, d_forward], dtype=np.float32),
            d_left=d_left,
            d_right=d_right,
            goal_distance=self._goal_distance(),
            collided=collided,
            off_lane=off_lane,
            lateral_deviation=lateral_dev,
        )
        return obs, info
