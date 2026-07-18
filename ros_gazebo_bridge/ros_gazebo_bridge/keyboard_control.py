"""Interactive keyboard teleop for the Gazebo+PX4+MAVROS backend.

The "fly it live with the keyboard" tool -- a companion to manual_control.py
(which runs a fixed scripted sequence). This one arms, enters OFFBOARD, and
then lets you pilot the drone in real time. It is the *perfect-control* file:
plain manual piloting, no STAR-Nav policy. (The paper/policy-driven flight
lives in a separate research-control file.)

Controls (hold-to-move; the drone brakes to a hover when no key is pressed):
    w / s   : PITCH forward / back   (+x / -x body velocity)
    a / d   : ROLL  left / right      (+y / -y body velocity)
    q / e   : YAW   left / right      (turn)
    r / f   : throttle UP / DOWN      (climb / descend)
    space   : immediate hover (zero all velocities, hold altitude)
    t       : take off (arm + OFFBOARD + climb to --takeoff-alt)
    l       : land + disarm
    x / Ctrl-C : quit (disarms)

So WASD are the roll/pitch (translation) axes, q/e is yaw, r/f is throttle --
i.e. a normal RC-style stick mapping on the keyboard.

Frames: velocities are sent in the BODY frame (FRAME_BODY_NED) so "forward"
is always where the nose points, regardless of heading. Altitude is held by
a position-Z lock that r/f nudges up/down, so the drone keeps its height
when you are only translating/yawing.

MUST be run in an interactive terminal (it puts the TTY in raw mode):
    docker exec -it star_nav_mavros bash -lc '
      export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
      source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
      python3 -m ros_gazebo_bridge.keyboard_control'
"""
from __future__ import annotations

import argparse
import select
import sys
import termios
import threading
import time
import tty

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import CommandBool, SetMode


# type_mask: velocity xyz + yaw_rate (ignore pos/acc/yaw)
_VEL_YAWRATE = (1 | 2 | 4 | 64 | 128 | 256 | 1024)
# type_mask: velocity xy + position z + yaw_rate (locks altitude)
_VELXY_POSZ_YAWRATE = (1 | 2 | 32 | 64 | 128 | 256 | 1024)

SETPOINT_HZ = 50.0


class KeyboardTeleop(Node):
    def __init__(self, args):
        super().__init__("star_nav_keyboard_control")
        self.args = args

        sensor_qos = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=QoSHistoryPolicy.KEEP_LAST, depth=10)
        self._state = State()
        self._pose = None
        self.create_subscription(State, args.state_topic, self._on_state, 10)
        self.create_subscription(PoseStamped, args.pose_topic, self._on_pose, sensor_qos)

        self._sp_pub = self.create_publisher(PositionTarget, args.setpoint_topic, 10)
        self._arm_cli = self.create_client(CommandBool, args.arm_service)
        self._mode_cli = self.create_client(SetMode, args.mode_service)

        # Commanded state (updated by keypresses, streamed by the timer).
        # Velocities are in the LOCAL ENU frame: +x = East ("forward" at
        # spawn heading), +y = North ("left"), altitude held via position-z.
        self._lock = threading.Lock()
        self._vx = self._vy = self._yaw_rate = 0.0
        self._cmd_alt = args.takeoff_alt
        self._mode = "idle"          # idle | fly | land
        self._want_offboard = False
        self._last_offboard_req = 0.0

        self.create_timer(1.0 / SETPOINT_HZ, self._stream_tick)
        self._executor = MultiThreadedExecutor(num_threads=3)
        self._executor.add_node(self)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

    # ---- callbacks -------------------------------------------------------
    def _on_state(self, msg: State):
        self._state = msg

    def _on_pose(self, msg: PoseStamped):
        self._pose = msg

    def _z(self):
        return self._pose.pose.position.z if self._pose is not None else 0.0

    def _stream_tick(self):
        with self._lock:
            vx, vy, yr, alt, mode = (self._vx, self._vy, self._yaw_rate,
                                     self._cmd_alt, self._mode)
        sp = PositionTarget()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        if mode == "fly":
            # velocity in xy, altitude LOCKED via position-z, plus yaw rate.
            sp.type_mask = _VELXY_POSZ_YAWRATE
            sp.velocity.x, sp.velocity.y = vx, vy
            sp.position.z = alt
            sp.yaw_rate = yr
        elif mode == "land":
            sp.type_mask = _VEL_YAWRATE
            sp.velocity.z = -0.4  # ENU: descend
        else:  # idle (pre-takeoff): hold zero velocity so OFFBOARD can arm
            sp.type_mask = _VEL_YAWRATE
        self._sp_pub.publish(sp)

        if (self._want_offboard and self._state.connected
                and self._state.mode != "OFFBOARD"):
            now = time.monotonic()
            if now - self._last_offboard_req > 0.5:
                self._last_offboard_req = now
                req = SetMode.Request()
                req.custom_mode = "OFFBOARD"
                self._mode_cli.call_async(req)

    # ---- commands --------------------------------------------------------
    def _call(self, cli, req, timeout=3.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            return None
        fut = cli.call_async(req)
        end = time.monotonic() + timeout
        while rclpy.ok() and not fut.done() and time.monotonic() < end:
            time.sleep(0.02)
        return fut.result() if fut.done() else None

    def takeoff(self):
        if self._mode != "idle":
            return
        self.get_logger().info("Taking off: pre-stream -> OFFBOARD -> arm ...")
        self._want_offboard = True
        time.sleep(0.5)  # let the stream establish
        m = SetMode.Request(); m.custom_mode = "OFFBOARD"
        self._call(self._mode_cli, m)
        a = CommandBool.Request(); a.value = True
        self._call(self._arm_cli, a)
        with self._lock:
            self._cmd_alt = self.args.takeoff_alt
            self._mode = "fly"
        self.get_logger().info(
            f"Airborne target {self.args.takeoff_alt:.1f} m -- "
            f"mode={self._state.mode} armed={self._state.armed}")

    def land(self):
        if self._mode == "idle":
            return
        self.get_logger().info("Landing ...")
        with self._lock:
            self._vx = self._vy = self._yaw_rate = 0.0
            self._mode = "land"   # timer now streams a descent velocity
        end = time.monotonic() + self.args.takeoff_alt / 0.4 + 3.0
        while rclpy.ok() and time.monotonic() < end and self._z() > 0.3:
            time.sleep(0.1)
        self._want_offboard = False
        with self._lock:
            self._mode = "idle"
        a = CommandBool.Request(); a.value = False
        self._call(self._arm_cli, a)
        self.get_logger().info("Disarmed.")


HELP = (
    "\r\n[keyboard teleop] w/s pitch(fwd/back)  a/d roll(left/right)  "
    "q/e yaw  r/f up/down  space hover  t takeoff  l land  x quit\r\n")


def get_key(timeout: float) -> str:
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if r else ""


def run_teleop(node: KeyboardTeleop):
    a = node.args
    v, climb, yaw = a.move_speed, a.climb_step, a.yaw_rate
    sys.stdout.write(HELP)
    sys.stdout.flush()
    while rclpy.ok():
        k = get_key(0.1).lower()
        if k == "":
            # no key this tick -> brake to hover (still holding altitude)
            with node._lock:
                node._vx = node._vy = node._yaw_rate = 0.0
            continue
        if k == "x":
            break
        if k == "t":
            node.takeoff(); continue
        if k == "l":
            node.land(); continue
        with node._lock:
            if k == "w":   node._vx = v
            elif k == "s": node._vx = -v
            elif k == "a": node._vy = v      # body +y = left
            elif k == "d": node._vy = -v
            elif k == "q": node._yaw_rate = yaw
            elif k == "e": node._yaw_rate = -yaw
            elif k == "r": node._cmd_alt += climb
            elif k == "f": node._cmd_alt = max(0.3, node._cmd_alt - climb)
            elif k == " ": node._vx = node._vy = node._yaw_rate = 0.0


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setpoint-topic", default="/mavros/mavros/local")
    p.add_argument("--arm-service", default="/mavros/mavros/arming")
    p.add_argument("--mode-service", default="/mavros/set_mode")
    p.add_argument("--state-topic", default="/mavros/state")
    p.add_argument("--pose-topic", default="/mavros/mavros/pose")
    p.add_argument("--takeoff-alt", type=float, default=2.5)
    p.add_argument("--move-speed", type=float, default=1.2, help="translate speed (m/s)")
    p.add_argument("--climb-step", type=float, default=0.3, help="altitude change per r/f press (m)")
    p.add_argument("--yaw-rate", type=float, default=0.6, help="yaw rate (rad/s)")
    return p


def main():
    args = build_argparser().parse_args()
    if not sys.stdin.isatty():
        print("keyboard_control needs an interactive TTY -- run with `docker exec -it`.")
        return
    rclpy.init()
    node = KeyboardTeleop(args)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        run_teleop(node)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        try:
            node.land()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
