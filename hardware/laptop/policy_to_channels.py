"""policy_to_channels.py -- STAGE 4 glue: send a STAR-Nav policy action out the
SAME rc_link the keyboard used, so the drone flies autonomously through the
Arduino -> trainer-port -> radio chain.

The AGSS-PPO policy emits a 4-D action a = [v_x, v_y, v_z, yaw_rate], each in
[-1, 1] (see star_nav/models/agss_ppo.py). On a real Betaflight/iNav radio in
ANGLE mode the sticks are roll/pitch/yaw/throttle, so the natural mapping is:

    v_x (forward)  -> pitch      v_y (lateral) -> roll
    yaw_rate       -> yaw        v_z (up/down) -> throttle (nudges around hover)

so this file is a thin adapter: policy action -> RCLink.send_norm().

!!! IMPORTANT CAVEAT (read before trusting this) !!!
The policy was TRAINED to output body VELOCITIES fed to a position/velocity
controller (PX4 offboard + altitude-hold). A stock FPV radio has no such
controller -- the sticks command ANGLE/RATE directly, and throttle is raw
thrust with no altitude hold. So this mapping is a *starting scaffold*, NOT a
drop-in: expect to (a) retrain/fine-tune the policy against an angle/rate action
space, or (b) add an onboard/laptop-side velocity->angle controller between the
policy and this adapter. This is exactly the sim2real gap discussed in the
project's deploy notes. Keep PROPS OFF and the trainer-switch override on.

Usage sketch (pseudo -- wire up your own perception loop):
    from rc_link import RCLink
    link = RCLink("/dev/ttyUSB0")
    send = make_sender(link, hover_throttle=0.15)
    while flying:
        action = policy_step(obs)          # np.array([vx, vy, vz, yaw]) in [-1,1]
        send(action, armed=True)
"""
from __future__ import annotations


def make_sender(link, hover_throttle=0.0, gains=(1.0, 1.0, 0.5, 1.0)):
    """Return send(action, armed) that maps a 4-D policy action to RC sticks.

    hover_throttle: normalized throttle [-1,1] the vehicle roughly hovers at
                    (v_z nudges around this). Measure it in manual flight first.
    gains:          (roll, pitch, yaw, vz) scaling to tame the raw policy output.
    """
    gr, gp, gy, gz = gains

    def send(action, armed=True):
        vx, vy, vz, yaw = (list(action) + [0, 0, 0, 0])[:4]
        link.send_norm(
            roll=gr * vy,
            pitch=gp * vx,
            yaw=gy * yaw,
            throttle=hover_throttle + gz * vz,
            arm=armed,
        )

    return send
