"""keyboard_control.py -- STAGE 1: fly the real drone manually from the laptop
keyboard, through the Arduino -> trainer-port -> radio -> drone chain.

Same key mapping as the sim tool (ros_gazebo_bridge/keyboard_control.py) so the
muscle memory carries over, but here the output goes to the Arduino via rc_link
instead of MAVROS. This is the file you use to PROVE the whole control chain
works (channel directions, arming, trim) with a human in the loop, BEFORE
handing the same rc_link over to the STAR-Nav policy (see policy_to_channels.py).

Controls:
    w / s   : PITCH forward / back
    a / d   : ROLL  left / right
    q / e   : YAW   left / right
    r / f   : THROTTLE up / down   (sticky -- holds where you leave it)
    space   : centre roll/pitch/yaw (throttle unchanged)
    t       : ARM      (arm channel high)
    g       : DISARM   (arm channel low, throttle to min)
    x / Ctrl-C : quit  (disarms)

roll/pitch/yaw self-centre when you're not pressing them (like releasing a
spring stick); throttle stays put. Streams at 50 Hz so the Arduino failsafe
never trips while you fly.

!!! PROPS OFF for the first run. Keep the radio trainer switch as override. !!!

    python keyboard_control.py --port /dev/ttyUSB0
"""
from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty

from rc_link import RCLink, US_MIN, US_MID, US_MAX, norm_to_us

RATE_HZ = 50.0
STEP = 0.08          # how fast a held key pushes the stick toward full
DECAY = 0.80         # roll/pitch/yaw relax toward centre each tick when idle
THR_STEP = 0.02      # throttle increment per r/f tick


def read_keys():
    """Non-blocking: return the set of chars available this instant."""
    keys = set()
    while select.select([sys.stdin], [], [], 0)[0]:
        keys.add(sys.stdin.read(1))
    return keys


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    args = p.parse_args(argv)

    link = RCLink(args.port, args.baud)
    roll = pitch = yaw = 0.0     # normalized [-1,1], self-centering
    thr = -1.0                   # normalized [-1,1]; -1 = min throttle
    armed = False

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    dt = 1.0 / RATE_HZ
    try:
        print("keyboard control: PROPS OFF first. t=arm g=disarm  wasd/qe/rf  x=quit", flush=True)
        while True:
            keys = read_keys()
            if "x" in keys:
                break
            if "t" in keys: armed = True
            if "g" in keys: armed = False; thr = -1.0
            if " " in keys: roll = pitch = yaw = 0.0

            # translation / yaw (self-centering)
            if "w" in keys: pitch = min(1.0, pitch + STEP)
            elif "s" in keys: pitch = max(-1.0, pitch - STEP)
            else: pitch *= DECAY
            if "d" in keys: roll = min(1.0, roll + STEP)
            elif "a" in keys: roll = max(-1.0, roll - STEP)
            else: roll *= DECAY
            if "e" in keys: yaw = min(1.0, yaw + STEP)
            elif "q" in keys: yaw = max(-1.0, yaw - STEP)
            else: yaw *= DECAY
            # throttle (sticky)
            if "r" in keys: thr = min(1.0, thr + THR_STEP)
            if "f" in keys: thr = max(-1.0, thr - THR_STEP)

            link.send_norm(roll=roll, pitch=pitch, yaw=yaw, throttle=thr, arm=armed)
            sys.stdout.write(f"\r arm={armed}  thr={thr:+.2f}  roll={roll:+.2f} "
                             f"pitch={pitch:+.2f} yaw={yaw:+.2f}   ")
            sys.stdout.flush()
            time.sleep(dt)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        link.close()
        print("\ndisarmed, link closed.")


if __name__ == "__main__":
    main()
