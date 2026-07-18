"""rc_link.py -- the ONE communication layer between the laptop and the drone.

Everything upstream (keyboard now, STAR-Nav policy later) just produces channel
values and calls RCLink.send_*(); this packs them into the 19-byte serial packet
ppm_trainer.ino expects and streams them to the Arduino. Swapping "keyboard" for
"policy" changes nothing here -- same 4-8 channels, same wire.

Channel order (must match ppm_trainer.ino's `failsafe[]` AND your EdgeTX mixer):
    0=roll  1=pitch  2=throttle  3=yaw  4=arm  5..7=aux      (AETR + arm)

Run a quick self-test (prints packet bytes, no serial) with:  python rc_link.py --selftest
"""
from __future__ import annotations

import struct
import time

HDR = b"\xA5\x5A"
N = 8
US_MIN, US_MID, US_MAX = 1000, 1500, 2000


def clamp_us(v):
    return int(max(US_MIN, min(US_MAX, round(v))))


def norm_to_us(x, lo=US_MIN, hi=US_MAX):
    """Map a normalized stick value [-1, 1] to microseconds [lo, hi] (0 -> centre)."""
    x = max(-1.0, min(1.0, float(x)))
    return int(round((x + 1.0) * 0.5 * (hi - lo) + lo))


def pack(channels):
    """channels: iterable of up to N microsecond values -> 19-byte packet."""
    ch = (list(channels) + [US_MID] * N)[:N]
    payload = b"".join(struct.pack("<H", clamp_us(v)) for v in ch)
    ck = 0
    for byte in payload:
        ck ^= byte
    return HDR + payload + bytes([ck])


class RCLink:
    def __init__(self, port="/dev/ttyUSB0", baud=115200):
        import serial  # pyserial; only needed when actually talking to hardware
        self.ser = serial.Serial(port, baud, timeout=0)
        time.sleep(2.0)  # Arduino auto-resets on serial open; wait for its setup()

    def send_us(self, channels):
        self.ser.write(pack(channels))

    def send_norm(self, roll=0.0, pitch=0.0, yaw=0.0, throttle=-1.0, arm=False, aux=None):
        """Convenience: normalized sticks [-1,1] (throttle too) + arm bool -> channels."""
        ch = [
            norm_to_us(roll), norm_to_us(pitch), norm_to_us(throttle), norm_to_us(yaw),
            US_MAX if arm else US_MIN,
        ]
        ch += list(aux) if aux else [US_MID, US_MID, US_MID]
        self.send_us(ch)

    def disarm(self):
        # throttle min, arm low -- what the failsafe would do anyway.
        self.send_norm(throttle=-1.0, arm=False)

    def close(self):
        try:
            self.disarm()
        finally:
            self.ser.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true", help="print example packets, no serial needed")
    args = p.parse_args()
    if args.selftest:
        print("centre + disarmed:", pack([1500, 1500, 1000, 1500, 1000]).hex(" "))
        print("full-forward pitch:", pack([1500, 2000, 1000, 1500, 1000]).hex(" "))
        print("norm_to_us(-1,0,1):", norm_to_us(-1), norm_to_us(0), norm_to_us(1))
