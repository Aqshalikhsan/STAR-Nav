"""Manual (scripted) OFFBOARD control for the Gazebo+PX4+MAVROS backend.

This is the "drive it by hand before autonomy" tool: it arms the vehicle,
switches PX4 into OFFBOARD, and flies a fixed, human-authored sequence of
manoeuvres (takeoff -> hover -> forward flight -> yaw -> land) by streaming
`mavros_msgs/PositionTarget` setpoints. It deliberately does NOT load any
STAR-Nav policy -- it exists to prove the *control path*
(arm -> OFFBOARD -> setpoint -> visible motion) works end to end, so that
when the learned policy later moves the drone we know the plumbing beneath
it is sound.

Setpoint streaming architecture
-------------------------------
PX4 refuses to stay in OFFBOARD unless setpoints keep arriving at > 2 Hz. A
naive "publish inside the flight loop" design drops OFFBOARD the moment the
loop does anything slow (a service call, a GC pause, or just MAVROS lagging
under load) -- observed as a mid-flight fall back to ALTCTL. So here a
background ROS timer republishes the *current* setpoint at a fixed 50 Hz,
driven by a MultiThreadedExecutor spinning on its own thread, completely
decoupled from the main flight-sequence thread. The flight sequence only
ever updates `self._current_sp` and sleeps; the stream never lapses. The
same timer re-commands OFFBOARD if PX4 ever drops out of it while we still
want to be flying.

Why the topic/service names look "doubled"
------------------------------------------
On this ros-humble-mavros build, `mavros_node` hardcodes its own node
namespace to `mavros`, so the documented `/mavros/setpoint_raw/local`
setpoint input is actually at `/mavros/mavros/local` and the arming service
at `/mavros/mavros/arming` (verified via `ros2 node info /mavros/mavros`).
`set_mode` stays `/mavros/set_mode`. EKF pose is read from
`/mavros/mavros/pose`. All overridable via CLI flags.

Frames
------
MAVROS takes local setpoints in ROS ENU (x=East, y=North, z=Up) and
converts to PX4 NED, with `coordinate_frame=FRAME_LOCAL_NED`. So `+x`
velocity is eastward, `+z` is up. The corridor "forward" axis in the
generated world is world +x, which maps onto local +x here.

Usage (inside the ros-bridge container, MAVROS already running):
    python3 -m ros_gazebo_bridge.manual_control            # default demo
    python3 -m ros_gazebo_bridge.manual_control --takeoff-alt 2.5 \
        --forward-speed 1.5 --forward-time 12 --no-land
"""
from __future__ import annotations

import argparse
import math
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, SetMode


# PositionTarget.type_mask ignore-bit combinations.
# Bits: PX=1 PY=2 PZ=4 | VX=8 VY=16 VZ=32 | AFX=64 AFY=128 AFZ=256 |
#       FORCE=512 | YAW=1024 | YAW_RATE=2048
_USE_POSITION_YAW = (8 | 16 | 32 | 64 | 128 | 256 | 2048)     # pos xyz + yaw
_USE_VELOCITY_YAWRATE = (1 | 2 | 4 | 64 | 128 | 256 | 1024)   # vel xyz + yaw_rate
# Velocity in xy but POSITION hold in z (locks altitude during forward
# flight -- a pure vz "hold" command drifts because a forward-pitched
# multicopter loses vertical thrust; commanding pz instead pins altitude).
_USE_VELXY_POSZ_YAWRATE = (1 | 2 | 32 | 64 | 128 | 256 | 1024)

SETPOINT_HZ = 50.0        # background stream rate (well above PX4's >2 Hz gate)


class ManualController(Node):
    def __init__(self, args):
        super().__init__("star_nav_manual_control")
        self.args = args

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._state = State()
        self._pose = None  # geometry_msgs/PoseStamped
        self.create_subscription(State, args.state_topic, self._on_state, 10)
        self.create_subscription(PoseStamped, args.pose_topic, self._on_pose, sensor_qos)

        self._sp_pub = self.create_publisher(PositionTarget, args.setpoint_topic, 10)
        self._arm_cli = self.create_client(CommandBool, args.arm_service)
        self._mode_cli = self.create_client(SetMode, args.mode_service)

        # The setpoint the background timer republishes. Guarded by a lock so
        # the flight thread can swap it atomically. Starts as a safe idle
        # position-hold at origin; replaced with the real hold pose in
        # arm_and_offboard().
        self._sp_lock = threading.Lock()
        self._current_sp = self._make_position_sp(0.0, 0.0, args.takeoff_alt)
        self._want_offboard = False
        self._last_offboard_req = 0.0

        # Background 50 Hz setpoint stream + OFFBOARD keepalive.
        self.create_timer(1.0 / SETPOINT_HZ, self._stream_tick)

        # Spin callbacks/timers on a background MultiThreadedExecutor so the
        # main thread is free to run the (blocking) flight sequence.
        self._executor = MultiThreadedExecutor(num_threads=3)
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    # ---- callbacks -------------------------------------------------------
    def _on_state(self, msg: State):
        self._state = msg

    def _on_pose(self, msg: PoseStamped):
        self._pose = msg

    def _stream_tick(self):
        """50 Hz: republish the current setpoint (fresh stamp) and re-command
        OFFBOARD if PX4 dropped out of it while we still want to fly."""
        with self._sp_lock:
            sp = self._current_sp
        sp.header.stamp = self.get_clock().now().to_msg()
        self._sp_pub.publish(sp)

        if (self._want_offboard and self._state.connected
                and self._state.mode != "OFFBOARD"):
            now = time.monotonic()
            if now - self._last_offboard_req > 0.5:  # rate-limit re-requests
                self._last_offboard_req = now
                req = SetMode.Request()
                req.custom_mode = "OFFBOARD"
                self._mode_cli.call_async(req)  # fire-and-forget

    # ---- setpoint builders ----------------------------------------------
    def _make_position_sp(self, x, y, z, yaw=0.0) -> PositionTarget:
        sp = PositionTarget()
        sp.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        sp.type_mask = _USE_POSITION_YAW
        sp.position.x, sp.position.y, sp.position.z = x, y, z
        sp.yaw = yaw
        return sp

    def _make_velocity_sp(self, vx, vy, vz, yaw_rate=0.0) -> PositionTarget:
        sp = PositionTarget()
        sp.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        sp.type_mask = _USE_VELOCITY_YAWRATE
        sp.velocity.x, sp.velocity.y, sp.velocity.z = vx, vy, vz
        sp.yaw_rate = yaw_rate
        return sp

    def _make_velxy_posz_sp(self, vx, vy, z, yaw_rate=0.0) -> PositionTarget:
        sp = PositionTarget()
        sp.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        sp.type_mask = _USE_VELXY_POSZ_YAWRATE
        sp.velocity.x, sp.velocity.y = vx, vy
        sp.position.z = z
        sp.yaw_rate = yaw_rate
        return sp

    def _set_sp(self, sp: PositionTarget):
        with self._sp_lock:
            self._current_sp = sp

    # ---- helpers ---------------------------------------------------------
    def _pos(self):
        if self._pose is None:
            return None
        p = self._pose.pose.position
        return (p.x, p.y, p.z)

    def _hold(self, sp: PositionTarget, duration: float, label: str):
        """Command `sp` and hold it for `duration` wall-clock seconds, logging
        EKF pose once per second. The background timer keeps publishing it."""
        self._set_sp(sp)
        t0 = time.monotonic()
        last_log = -1.0
        while rclpy.ok() and time.monotonic() - t0 < duration:
            t = time.monotonic() - t0
            if t - last_log >= 1.0:
                last_log = t
                pos = self._pos()
                pstr = ("(%.2f, %.2f, %.2f)" % pos) if pos else "(no pose)"
                self.get_logger().info(
                    f"[{label}] t={t:4.1f}s pos={pstr} "
                    f"mode={self._state.mode} armed={self._state.armed}")
            time.sleep(0.05)

    def _track(self, make_sp, duration: float, label: str):
        """Like _hold but re-evaluates make_sp() each control step (for
        setpoints that depend on live pose)."""
        t0 = time.monotonic()
        last_log = -1.0
        while rclpy.ok() and time.monotonic() - t0 < duration:
            self._set_sp(make_sp())
            t = time.monotonic() - t0
            if t - last_log >= 1.0:
                last_log = t
                pos = self._pos()
                pstr = ("(%.2f, %.2f, %.2f)" % pos) if pos else "(no pose)"
                self.get_logger().info(
                    f"[{label}] t={t:4.1f}s pos={pstr} "
                    f"mode={self._state.mode} armed={self._state.armed}")
            time.sleep(0.05)

    # ---- flight sequence -------------------------------------------------
    def wait_for_connection(self, timeout=30.0):
        self.get_logger().info("Waiting for MAVROS FCU connection...")
        end = time.monotonic() + timeout
        while rclpy.ok() and not self._state.connected and time.monotonic() < end:
            time.sleep(0.1)
        if not self._state.connected:
            raise RuntimeError("MAVROS never reported connected=true")
        self.get_logger().info("FCU connected.")

        end = time.monotonic() + timeout
        while rclpy.ok() and self._pose is None and time.monotonic() < end:
            time.sleep(0.1)
        if self._pose is None:
            self.get_logger().warn(
                "No EKF pose yet -- proceeding with origin-relative setpoints.")

    def _call_service(self, cli, req, name, timeout=5.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {name} unavailable")
        fut = cli.call_async(req)
        end = time.monotonic() + timeout
        while rclpy.ok() and not fut.done() and time.monotonic() < end:
            time.sleep(0.02)
        if not fut.done():
            raise RuntimeError(f"service {name} call timed out")
        return fut.result()

    def arm_and_offboard(self):
        pos = self._pos() or (0.0, 0.0, 0.0)
        self._hold_x, self._hold_y = pos[0], pos[1]
        alt = self.args.takeoff_alt

        # Pre-stream the takeoff hold setpoint so PX4 will accept OFFBOARD.
        self._set_sp(self._make_position_sp(self._hold_x, self._hold_y, alt))
        self._want_offboard = True
        self.get_logger().info("Pre-streaming setpoints for OFFBOARD entry...")
        time.sleep(1.5)

        self.get_logger().info("Requesting OFFBOARD mode...")
        mode_req = SetMode.Request()
        mode_req.custom_mode = "OFFBOARD"
        res = self._call_service(self._mode_cli, mode_req, self.args.mode_service)
        self.get_logger().info(f"  set_mode sent (mode_sent={getattr(res,'mode_sent',None)})")

        self.get_logger().info("Arming...")
        arm_req = CommandBool.Request()
        arm_req.value = True
        res = self._call_service(self._arm_cli, arm_req, self.args.arm_service)
        self.get_logger().info(f"  arming (success={getattr(res,'success',None)})")

        time.sleep(2.0)
        self.get_logger().info(
            f"After arm/offboard: mode={self._state.mode} armed={self._state.armed}")
        if not self._state.armed or self._state.mode != "OFFBOARD":
            self.get_logger().warn(
                "Not armed+OFFBOARD as expected -- flight may not proceed. "
                "Check PX4 preflight (px4-commander check).")

    def fly(self):
        a = self.args
        alt = a.takeoff_alt
        hx, hy = self._hold_x, self._hold_y

        # 1) Climb to takeoff altitude and stabilise (position hold).
        self.get_logger().info(f"== TAKEOFF to {alt:.1f} m ==")
        self._hold(self._make_position_sp(hx, hy, alt), a.takeoff_time, "takeoff")

        # 2) Forward flight: velocity in +x, altitude LOCKED via position-z.
        self.get_logger().info(
            f"== FORWARD {a.forward_speed:.1f} m/s for {a.forward_time:.0f}s ==")
        self._hold(self._make_velxy_posz_sp(a.forward_speed, 0.0, alt),
                   a.forward_time, "forward")

        # 3) Hover to bleed off velocity (hold current position).
        pos = self._pos() or (hx, hy, alt)
        self.get_logger().info("== HOVER (hold current position) ==")
        self._hold(self._make_position_sp(pos[0], pos[1], alt), 3.0, "hover")

        if a.yaw_time > 0:
            self.get_logger().info(f"== YAW sweep {a.yaw_time:.0f}s ==")
            self._hold(self._make_velxy_posz_sp(0.0, 0.0, alt, math.radians(30)),
                       a.yaw_time, "yaw")
            pos = self._pos() or pos
            self._hold(self._make_position_sp(pos[0], pos[1], alt), 2.0, "settle")

        # 4) Land (descend), then disarm.
        if not a.no_land:
            self.get_logger().info("== LAND ==")
            self._hold(self._make_velocity_sp(0.0, 0.0, -0.5, 0.0),
                       alt / 0.5 + 2.0, "land")
            self._want_offboard = False
            self.disarm()
        else:
            self.get_logger().info("== HOLD (no-land) -- keeping OFFBOARD hover ==")
            self._hold(self._make_position_sp(pos[0], pos[1], alt), 5.0, "hold")

    def fly_all_axes(self):
        """Fly every basic direction so 6-DOF control is visibly exercised:
        hover, forward/back, left/right, up/down, yaw. Horizontal moves use a
        local-frame velocity setpoint with altitude LOCKED via position-z; the
        vehicle keeps yaw=0 throughout the translations, so local +x reads as
        'forward', +y as 'left' (yaw is demonstrated last). Between moves it
        holds zero velocity (hover in place)."""
        a = self.args
        alt = a.takeoff_alt
        hx, hy = self._hold_x, self._hold_y
        v = a.move_speed
        mt = a.move_time
        cmd_alt = alt

        def hover(dt, label="hover"):
            self._hold(self._make_velxy_posz_sp(0.0, 0.0, cmd_alt), dt, label)

        # 1) Takeoff + settle.
        self.get_logger().info(f"== TAKEOFF to {alt:.1f} m ==")
        self._hold(self._make_position_sp(hx, hy, alt), a.takeoff_time, "takeoff")
        self.get_logger().info("== HOVER ==")
        hover(3.0)

        # 2) Horizontal directions (each followed by a braking hover).
        for label, vx, vy in [
            ("FORWARD (+x)", v, 0.0),
            ("BACKWARD (-x)", -v, 0.0),
            ("LEFT (+y)", 0.0, v),
            ("RIGHT (-y)", 0.0, -v),
        ]:
            self.get_logger().info(f"== {label} {v:.1f} m/s x {mt:.0f}s ==")
            self._hold(self._make_velxy_posz_sp(vx, vy, cmd_alt), mt, label.split()[0])
            hover(2.0)

        # 3) Up then down (change the locked altitude).
        cmd_alt = alt + a.climb_step
        self.get_logger().info(f"== UP to {cmd_alt:.1f} m ==")
        hover(3.0, "UP")
        cmd_alt = alt
        self.get_logger().info(f"== DOWN to {cmd_alt:.1f} m ==")
        hover(3.0, "DOWN")

        # 4) Yaw right, then yaw back left.
        self.get_logger().info("== YAW right ==")
        self._hold(self._make_velxy_posz_sp(0.0, 0.0, cmd_alt, math.radians(30)), mt, "YAW+")
        hover(2.0)
        self.get_logger().info("== YAW left ==")
        self._hold(self._make_velxy_posz_sp(0.0, 0.0, cmd_alt, math.radians(-30)), mt, "YAW-")
        hover(2.0)

        # 5) Land or hold.
        if not a.no_land:
            self.get_logger().info("== LAND ==")
            self._hold(self._make_velocity_sp(0.0, 0.0, -0.5, 0.0),
                       cmd_alt / 0.5 + 2.0, "land")
            self._want_offboard = False
            self.disarm()
        else:
            self.get_logger().info("== HOLD (no-land) ==")
            hover(5.0, "hold")

    def disarm(self):
        self.get_logger().info("Disarming...")
        req = CommandBool.Request()
        req.value = False
        try:
            self._call_service(self._arm_cli, req, self.args.arm_service)
        except Exception as e:  # noqa: BLE001
            self.get_logger().warn(f"disarm failed: {e}")

    def shutdown(self):
        self._want_offboard = False
        try:
            self._executor.shutdown(timeout_sec=1.0)
        except Exception:  # noqa: BLE001
            pass


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setpoint-topic", default="/mavros/mavros/local")
    p.add_argument("--arm-service", default="/mavros/mavros/arming")
    p.add_argument("--mode-service", default="/mavros/set_mode")
    p.add_argument("--state-topic", default="/mavros/state")
    p.add_argument("--pose-topic", default="/mavros/mavros/pose")
    p.add_argument("--maneuver", choices=["allaxes", "corridor"], default="allaxes",
                   help="allaxes = fly every direction (hover/forward/back/left/right/"
                        "up/down/yaw); corridor = takeoff->forward->hover->land")
    p.add_argument("--takeoff-alt", type=float, default=2.5, help="target altitude (m)")
    p.add_argument("--takeoff-time", type=float, default=6.0, help="climb+settle time (s)")
    p.add_argument("--forward-speed", type=float, default=1.5, help="corridor +x velocity (m/s)")
    p.add_argument("--forward-time", type=float, default=8.0, help="corridor forward duration (s)")
    p.add_argument("--yaw-time", type=float, default=4.0, help="corridor yaw sweep duration (s), 0 to skip")
    # allaxes tuning
    p.add_argument("--move-speed", type=float, default=1.0, help="allaxes translate speed (m/s)")
    p.add_argument("--move-time", type=float, default=3.0, help="allaxes seconds per direction")
    p.add_argument("--climb-step", type=float, default=1.0, help="allaxes up/down step (m)")
    p.add_argument("--no-land", action="store_true", help="hover-hold instead of landing")
    return p


def main():
    args = build_argparser().parse_args()
    rclpy.init()
    node = ManualController(args)
    try:
        node.wait_for_connection()
        node.arm_and_offboard()
        if args.maneuver == "allaxes":
            node.fly_all_axes()
        else:
            node.fly()
        node.get_logger().info("Manual control sequence complete.")
    except Exception as e:  # noqa: BLE001
        node.get_logger().error(f"Manual control failed: {type(e).__name__}: {e}")
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
