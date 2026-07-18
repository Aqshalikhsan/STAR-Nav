"""rclpy node bridging PX4 (via MAVROS) and Gazebo (via ros_gz_bridge) into
the plain-Python request/response shape ``GazeboROSEnv`` needs.

Topic/service map (all names configurable under ``env.ros.*`` in
``configs/default.yaml``, defaults shown):

  Subscribed (sensor path -> EnvObservation, what a real drone would have):
    <imu_topic>   (default /mavros/mavros/data)          sensor_msgs/Imu             -> obs.imu
    <pose_topic>  (default /mavros/mavros/pose)          geometry_msgs/PoseStamped   -> obs.pose (EKF estimate, matches VIO pose in the paper)

  NOTE on the doubled-namespace defaults above: on this ros-humble-mavros
  build, mavros_node's C++ source hardcodes its own node namespace to
  "mavros" (independent of anything this package's launch file sets), and
  most plugins additionally build their topic names as relative
  "mavros/<name>" strings -- ROS 2 then prepends the node namespace on
  top, doubling it to "/mavros/mavros/<name>". A handful of topics (e.g.
  /mavros/state, /mavros/local_position/pose) are declared as absolute
  paths internally and so do NOT get doubled -- but they also turn out to
  have zero publishers on this build (only other MAVROS plugins subscribe
  to them internally, e.g. setpoint_position tracking current pose). The
  live, verified-flowing data is under the doubled paths. If you're on a
  different mavros build where this doesn't happen, override
  env.ros.imu_topic / env.ros.pose_topic to the plain (non-doubled) paths.
    <rgb_topic>                      sensor_msgs/Image           -> obs.rgb   (bridged from the vehicle's gz camera via ros_gz_bridge)
    <depth_topic>                    sensor_msgs/Image (32FC1)   -> depth thirds -> theta_corr_gt d_L/d_R/d_C / d_left / d_right
                                      (theta_corr_gt's phi component comes from gt_odom's yaw, not depth -- see env.py._observe)

  Subscribed (privileged path -> PrivilegedInfo only, never exposed to the policy):
    /mavros/state                    mavros_msgs/State           -> armed/connected sanity checks
    <gt_odom_topic>                  nav_msgs/Odometry           -> ground-truth xy for collision / goal_distance / lateral_deviation, and yaw -> theta_corr_gt phi
                                      (bridged from Gazebo's own odometry publisher for the vehicle model --
                                       NOT the MAVROS/EKF estimate, so privileged info matches the other two backends'
                                       ground-truth semantics exactly)

  Published:
    /mavros/mavros/local              mavros_msgs/PositionTarget  <- body-frame (v_x, v_y, v_z, yaw_rate) setpoints
                                       (doubled namespace on this build; the documented /mavros/setpoint_raw/local
                                        has zero subscribers -- see the setpoint_topic NOTE below)

  Services called:
    /mavros/mavros/arming             mavros_msgs/srv/CommandBool
    /mavros/set_mode                  mavros_msgs/srv/SetMode
    /world/<world_name>/set_pose       ros_gz_interfaces/srv/SetEntityPose   (teleport the vehicle back to the start pose on reset)
    /world/<world_name>/control        ros_gz_interfaces/srv/ControlWorld    (pause/reset physics on reset)

Segmentation: the fpv5/pavo_femto models carry a gz
``<sensor type="segmentation">`` publishing a per-pixel class-id label map
(``<seg_topic>``, default ``/semantic_segmentation/labels_map``), read into
``seg_mask`` below. Class ids follow mock_env.py's scheme
(SKY=0/GROUND=1/TRUNK=2/CANOPY=3/ACTOR=4) via world_gen.py's Label plugins.
Known coarseness: the oil_palm mesh can't be split per-part, so the whole
tree is TRUNK (CANOPY/ACTOR have no real pixels); and PX4's stock
x500_depth has no segmentation camera, so with that model (or if
``seg_topic`` is unset) ``seg_mask`` falls back to an all-``CLASS_UNKNOWN``
placeholder. See README.md's Limitations section. Depth-derived signals
(theta_corr_gt d_L/d_R/d_C, from the depth camera) are real and do not
depend on segmentation.
"""
from __future__ import annotations

import threading
import time

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.executors import MultiThreadedExecutor
    from sensor_msgs.msg import Image, Imu
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from mavros_msgs.msg import State, PositionTarget
    from mavros_msgs.srv import CommandBool, SetMode
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ros_bridge_node requires a sourced ROS 2 install with rclpy, "
        "sensor_msgs, geometry_msgs, nav_msgs and mavros_msgs on the Python "
        "path (i.e. `source /opt/ros/<distro>/setup.bash` plus MAVROS). "
        "See ros_gazebo_bridge/README.md, or use env.name: mock / env.name: "
        "airsim in configs/default.yaml to avoid this dependency entirely."
    ) from exc

try:
    from cv_bridge import CvBridge
except ImportError as exc:  # pragma: no cover
    raise ImportError("ros_bridge_node requires cv_bridge (ros-<distro>-cv-bridge).") from exc

CLASS_UNKNOWN = 0

SETPOINT_HZ = 50.0  # background setpoint stream rate (well above PX4's >2 Hz OFFBOARD gate)
OFFBOARD_PRESTREAM_STEPS = 20  # PX4 requires a >=2 Hz setpoint stream before it will accept OFFBOARD


class ROSGazeboBridge(Node):
    def __init__(self, ros_cfg, node_name: str = "star_nav_bridge"):
        super().__init__(node_name)
        self.ros_cfg = ros_cfg
        self._bridge = CvBridge()

        self._rgb = None
        self._depth = None
        self._seg = None
        self._imu = None
        self._local_pose = None   # EKF-estimated pose (goes into EnvObservation)
        self._gt_odom = None      # ground-truth pose (PrivilegedInfo only)
        self._mavros_state = None

        # All subscriptions share a ReentrantCallbackGroup so a slow
        # callback can't block the others. This matters specifically for
        # the imu/pose topics: MAVROS publishes them BEST_EFFORT (no
        # buffering/retry), while the RGB/depth callbacks do relatively
        # expensive cv_bridge image conversions. Under the default
        # (mutually-exclusive) callback group -- even with a
        # MultiThreadedExecutor -- callbacks still run one at a time, so
        # the image conversions serialize ahead of the imu/pose callbacks
        # and their best-effort messages get dropped before ever being
        # delivered. This was observed as GazeboROSEnv.reset() timing out
        # on *only* imu/local_pose while every reliable topic (rgb, depth,
        # gt_odom, state) came through. Reentrant group + the
        # MultiThreadedExecutor set up in env.py lets the image and
        # imu/pose callbacks actually run concurrently.
        self._cb_group = ReentrantCallbackGroup()
        qos = qos_profile_sensor_data
        cbg = self._cb_group

        def _sub(msg_type, topic, cb):
            return self.create_subscription(msg_type, topic, cb, qos, callback_group=cbg)

        _sub(Image, ros_cfg.rgb_topic, self._on_rgb)
        _sub(Image, ros_cfg.depth_topic, self._on_depth)
        seg_topic = getattr(ros_cfg, "seg_topic", None)
        if seg_topic:
            _sub(Image, seg_topic, self._on_seg)
        imu_topic = getattr(ros_cfg, "imu_topic", None) or "/mavros/mavros/data"
        pose_topic = getattr(ros_cfg, "pose_topic", None) or "/mavros/mavros/pose"
        _sub(Imu, imu_topic, self._on_imu)
        _sub(PoseStamped, pose_topic, self._on_local_pose)
        _sub(Odometry, ros_cfg.gt_odom_topic, self._on_gt_odom)
        _sub(State, "/mavros/state", self._on_state)

        # Service/topic names are configurable because this MAVROS build
        # exposes some of them under a doubled `/mavros/mavros/` namespace
        # (same quirk as the imu/pose topics -- see the NOTE above). The
        # arming service in particular is `/mavros/mavros/arming` here, NOT
        # the documented `/mavros/cmd/arming` (which doesn't exist on this
        # build, so reset()/arm_and_offboard() timed out waiting for it).
        # set_mode is NOT doubled. Verified via `ros2 node info /mavros/mavros`.
        # The setpoint_raw/local input is likewise doubled: MAVROS's
        # setpoint_raw plugin subscribes to `/mavros/mavros/local`
        # (mavros_msgs/PositionTarget), NOT `/mavros/setpoint_raw/local`
        # (which has zero subscribers on this build) -- publishing to the
        # documented name silently dropped every setpoint, so step()'s actions
        # never reached PX4 and the vehicle never moved. Verified via
        # `ros2 node info /mavros/mavros` (the PositionTarget subscription).
        setpoint_topic = getattr(ros_cfg, "setpoint_topic", None) or "/mavros/mavros/local"
        arm_service = getattr(ros_cfg, "arm_service", None) or "/mavros/mavros/arming"
        mode_service = getattr(ros_cfg, "mode_service", None) or "/mavros/set_mode"
        self._setpoint_pub = self.create_publisher(PositionTarget, setpoint_topic, 10)
        self._arm_client = self.create_client(CommandBool, arm_service, callback_group=cbg)
        self._mode_client = self.create_client(SetMode, mode_service, callback_group=cbg)

        # Spin the node on a background MultiThreadedExecutor thread rather
        # than driving rclpy.spin_once() inline from the env's step loop.
        # With the ReentrantCallbackGroup above this lets callbacks run
        # concurrently (see the note there), and it decouples message
        # processing from the env's step timing -- spin_for()/
        # wait_until_ready() below just poll the cached state while this
        # thread keeps delivering messages and service responses.
        self._executor = MultiThreadedExecutor(num_threads=6)
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        # Continuous background setpoint stream. PX4 drops OFFBOARD the instant
        # setpoints stop arriving at >2 Hz; step() only issues one setpoint per
        # step and the slow cv_bridge image decode between steps can exceed
        # that gap, so a dedicated 50 Hz timer republishes the latest commanded
        # setpoint (and re-commands OFFBOARD if PX4 ever drops it) independently
        # of step() timing. send_body_velocity_setpoint() just updates
        # _current_setpoint. (Same fix verified in manual_control.py.)
        self._sp_lock = threading.Lock()
        self._current_setpoint = self._build_body_vel_setpoint(0.0, 0.0, 0.0, 0.0)
        self._want_offboard = False
        self._last_offboard_req = 0.0
        self.create_timer(1.0 / SETPOINT_HZ, self._stream_tick, callback_group=cbg)

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------
    def _on_rgb(self, msg: Image) -> None:
        self._rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _on_depth(self, msg: Image) -> None:
        self._depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def _on_seg(self, msg: Image) -> None:
        # gz's semantic-segmentation labels_map is published as an rgb8 image
        # with the integer class id replicated across all three channels
        # (R == G == B == label; verified live: unique values {0,1,2} =
        # SKY/GROUND/TRUNK on every channel). Convert to rgb8 and take one
        # channel as the class-id map. Do NOT request mono8: that runs an
        # RGB->gray luminance conversion which only preserves the id by
        # coincidence (the 0.299/0.587/0.114 weights sum to 1 and the
        # channels are equal) -- a single channel is exact and self-evident.
        # desired_encoding="rgb8" also cleanly handles a genuinely single-
        # channel seg source (mono8 -> rgb8 replicates, channel 0 == value).
        img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self._seg = img[:, :, 0].astype(np.int64)

    def _on_imu(self, msg: Imu) -> None:
        self._imu = np.array([
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
        ], dtype=np.float32)

    def _on_local_pose(self, msg: PoseStamped) -> None:
        p, o = msg.pose.position, msg.pose.orientation
        self._local_pose = np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float32)

    def _on_gt_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        self._gt_odom = np.array([p.x, p.y, p.z, o.x, o.y, o.z, o.w], dtype=np.float32)

    def _on_state(self, msg: State) -> None:
        self._mavros_state = msg

    # ------------------------------------------------------------------
    # Blocking helpers used by GazeboROSEnv
    # ------------------------------------------------------------------
    def spin_for(self, seconds: float) -> None:
        # The background executor thread (see __init__) delivers messages
        # continuously, so this just lets wall-clock time pass -- it does
        # NOT drive rclpy itself (spinning the node from here as well as
        # from the executor thread would be a double-spin error).
        time.sleep(max(0.0, seconds))

    def wait_until_ready(self, timeout_sec: float = 30.0) -> None:
        """Blocks until at least one message has arrived on every required
        topic, so the first observation returned by reset() is never
        built from stale/None fields.
        """
        deadline = time.monotonic() + timeout_sec
        required = ["_rgb", "_depth", "_imu", "_local_pose", "_gt_odom", "_mavros_state"]
        while time.monotonic() < deadline:
            if all(getattr(self, name) is not None for name in required):
                return
            time.sleep(0.05)
        missing = [name.lstrip("_") for name in required if getattr(self, name) is None]
        raise TimeoutError(
            f"Timed out waiting for initial sensor data; never received: {missing}. "
            "Check that PX4 SITL, MAVROS, and ros_gz_bridge are all running "
            "(see ros_gazebo_bridge/README.md)."
        )

    def _build_body_vel_setpoint(self, v_x: float, v_y: float, v_z: float, yaw_rate: float) -> PositionTarget:
        msg = PositionTarget()
        msg.header.frame_id = "base_link"
        msg.coordinate_frame = PositionTarget.FRAME_BODY_NED
        msg.type_mask = (
            PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ
            | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ
            | PositionTarget.IGNORE_YAW
        )
        msg.velocity.x = v_x
        msg.velocity.y = v_y
        msg.velocity.z = v_z
        msg.yaw_rate = yaw_rate
        return msg

    def send_body_velocity_setpoint(self, v_x: float, v_y: float, v_z: float, yaw_rate: float) -> None:
        # Update the setpoint the 50 Hz background timer republishes -- do NOT
        # publish once here (a per-step publish lapses OFFBOARD between slow
        # steps; the timer keeps the stream alive).
        sp = self._build_body_vel_setpoint(v_x, v_y, v_z, yaw_rate)
        with self._sp_lock:
            self._current_setpoint = sp

    def _build_local_pos_setpoint(self, x: float, y: float, z: float, yaw: float) -> PositionTarget:
        """A position setpoint in the MAVROS local ENU frame (FRAME_LOCAL_NED is
        the MAVLink enum name; MAVROS interprets/publishes local_position in ENU
        and transforms to PX4's NED internally, so x/y/z here are the SAME frame
        as /mavros/mavros/pose). Used by the inference-follower deploy
        (scripts/fly_inference_gazebo.py) to position-track a Mock-exported path
        instead of running the wandering live policy."""
        msg = PositionTarget()
        msg.header.frame_id = "map"
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = (
            PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ
            | PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ
            | PositionTarget.IGNORE_YAW_RATE
        )
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = float(z)
        msg.yaw = float(yaw)
        return msg

    def send_local_position_setpoint(self, x: float, y: float, z: float, yaw: float = 0.0,
                                     vx: float = None, vy: float = None) -> None:
        """Command an absolute position (MAVROS local ENU) the 50 Hz stream holds
        until the next call. Optionally pass a horizontal velocity FEED-FORWARD
        (vx, vy) in the same frame: PX4 then adds it to the position-error output
        instead of deriving all velocity from position error, which makes tracking
        markedly smoother (less attitude twitch) along a moving carrot."""
        sp = self._build_local_pos_setpoint(x, y, z, yaw)
        if vx is not None and vy is not None:
            sp.type_mask &= ~(PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY)
            sp.velocity.x = float(vx)
            sp.velocity.y = float(vy)
        with self._sp_lock:
            self._current_setpoint = sp

    def local_xy(self):
        """Current EKF local position (ENU x, y) or None if no pose yet."""
        return None if self._local_pose is None else self._local_pose[:2].copy()

    def gt_xy(self):
        """Current ground-truth world position (x, y) or None."""
        return None if self._gt_odom is None else self._gt_odom[:2].copy()

    def _stream_tick(self) -> None:
        with self._sp_lock:
            sp = self._current_setpoint
        sp.header.stamp = self.get_clock().now().to_msg()
        self._setpoint_pub.publish(sp)
        state = getattr(self, "_mavros_state", None)
        if (self._want_offboard and state is not None
                and getattr(state, "connected", False)
                and state.mode != "OFFBOARD"):
            now = time.monotonic()
            if now - self._last_offboard_req > 0.5:  # rate-limit re-requests
                self._last_offboard_req = now
                req = SetMode.Request()
                req.custom_mode = "OFFBOARD"
                self._mode_client.call_async(req)

    def _call_service_sync(self, client, request, timeout_sec: float = 5.0):
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise TimeoutError(f"Service {client.srv_name} not available after {timeout_sec}s")
        future = client.call_async(request)
        # The background executor thread processes the service response, so
        # poll future.done() rather than spin_until_future_complete (which
        # would double-spin the node against that thread).
        deadline = time.monotonic() + timeout_sec
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"Service call to {client.srv_name} timed out or failed")
        return future.result()

    def arm_and_offboard(self) -> None:
        """PX4 rejects an OFFBOARD mode switch unless it has already been
        receiving setpoints at >=2 Hz for a short period, so pre-stream a
        burst of zero-velocity setpoints before requesting the mode change.
        """
        # The background timer is already republishing _current_setpoint (zero
        # velocity) continuously, satisfying PX4's pre-OFFBOARD >2 Hz gate;
        # just make sure it is the zero-vel hold and let it establish.
        self.send_body_velocity_setpoint(0.0, 0.0, 0.0, 0.0)
        self._want_offboard = True
        time.sleep(OFFBOARD_PRESTREAM_STEPS / SETPOINT_HZ)

        mode_req = SetMode.Request()
        mode_req.custom_mode = "OFFBOARD"
        self._call_service_sync(self._mode_client, mode_req)

        arm_req = CommandBool.Request()
        arm_req.value = True
        self._call_service_sync(self._arm_client, arm_req)

    def disarm(self) -> None:
        self._want_offboard = False
        arm_req = CommandBool.Request()
        arm_req.value = False
        try:
            self._call_service_sync(self._arm_client, arm_req)
        except (TimeoutError, RuntimeError):
            pass  # best-effort on episode teardown

    def get_latest(self):
        """Returns the most recently received (rgb, depth, seg, imu,
        local_pose, gt_pose) tuple. Callers are responsible for calling
        spin_for()/wait_until_ready() first so these are fresh.
        """
        seg = self._seg if self._seg is not None else np.full(self._depth.shape, CLASS_UNKNOWN, dtype=np.int64)
        return self._rgb, self._depth, seg, self._imu, self._local_pose, self._gt_odom

    def shutdown(self) -> None:
        """Stops the background executor thread. Call before destroy_node()
        (GazeboROSEnv.close() does this) so the spin thread exits cleanly."""
        try:
            self._executor.shutdown()
        except Exception:
            pass
        if self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
