# STAR-Nav real-hardware bridge (laptop → Arduino → radio → drone)

Ground-side control bridge for flying a real FPV drone (Pavo Femto / fpv5-class)
from the laptop when the flight controller runs **Betaflight/iNav** (no PX4
offboard). The laptop's channel values are injected into the radio through its
**trainer port** via an Arduino generating PPM; the radio then transmits them
over its normal **ELRS/RF** link. No companion computer on the drone — all the
compute (perception + policy) runs on the laptop.

Design goal: **manual keyboard and the STAR-Nav policy use the exact same
`rc_link`** — prove the chain by hand first, then swap in the policy with zero
wiring change.

---

## Two control modes (same wire, different "brain")
```
[ keyboard_control.py ]──┐   MANUAL (WASD/qe/rf)
                         ├─▶ rc_link.py ─USB─▶ Arduino(ppm_trainer.ino) ─PPM(D9→TIP)─▶ radio TRAINER port ─ELRS─▶ drone
[ vision_deploy.py ]─────┘   POLICY (camera→SACR→CAMR→PPO)
```
| Mode | Entry point | Status |
|---|---|---|
| **1. Manual keyboard** | `laptop/keyboard_control.py` | ✅ complete, ready to run |
| **2. Policy (vision)** | `laptop/vision_deploy.py` (via `policy_to_channels.py`) | ✅ plumbing complete; needs sim2real tuning to fly well |

Run **one at a time** (both write the same serial/Arduino). Order: keyboard first
(verify chain + channel directions), then policy.

---

## Files
| File | What it is |
|---|---|
| `arduino/ppm_trainer/ppm_trainer.ino` | Arduino firmware: serial packets → 8-ch PPM on **D9**, with link-loss **failsafe** |
| `laptop/rc_link.py` | the one serial/comms layer (`send_us` / `send_norm`); everything upstream uses this |
| `laptop/keyboard_control.py` | **Mode 1** — manual flight from the keyboard |
| `laptop/policy_to_channels.py` | maps a policy action `[vx,vy,vz,yaw]` → RC sticks |
| `laptop/vision_deploy.py` | **Mode 2** — full loop: camera stream → SACR → CAMR → policy → `rc_link` |
| `config/channels.yaml` | channel order / endpoints / serial port / failsafe — keep in sync with the `.ino` **and** your radio |

---

## Full autonomous loop (Mode 2)
Two **separate wireless links** meet in the laptop; the drone carries nothing extra:
```
 DJI cam ─▶ goggles ─▶ Raspi(CosmoStream) ─RTSP/WiFi─▶  LAPTOP  ─USB─▶ Arduino ─PPM─▶ radio ─ELRS─▶ drone
   (video DOWN-link)                                 SACR→CAMR→policy→AGSS      (control UP-link)
```
Get frames into `vision_deploy.py` via **one** of:
* **RTSP straight from CosmoStream** (recommended, lowest latency): `--source rtsp://<pi-ip>:8554/cam`
* **OBS Virtual Camera** (only if you need overlay/recording): OBS → *Start Virtual Camera* → `--source 10` (the `/dev/videoN` index).

Every hop (goggles→Pi→WiFi→OBS→Python) adds control-loop latency — skip OBS if you can.

---

## Wiring (3 connections total)
```
Arduino USB  ───────────────▶ laptop            (power + serial packets)
Arduino D9   ──── wire ──────▶ 3.5mm jack TIP    (PPM signal)   ── into radio TRAINER port
Arduino GND  ──── wire ──────▶ 3.5mm jack SLEEVE (ground)       ──/
```
- The jack goes to the **Arduino pins (D9 + GND)**, *not* a USB port. Use a **3.5 mm aux cable with one end cut** (or a 3.5 mm breakout board).
- **Identify Tip vs Sleeve with a multimeter continuity test — do NOT trust wire colour** (in audio cables "red" is usually the unused Ring, not the signal). **Tip → D9**, **Sleeve → GND**, Ring → leave unconnected.
- 5 V Arduino PPM into a 3.3 V trainer port is usually fine; if unsure, add a ~1 kΩ series resistor on the signal wire.
- **First check your radio actually HAS a 3.5 mm trainer jack.** Big radios (RadioMaster TX16S, FrSky Taranis) do; small ones (RadioMaster Pocket/Zorro) often don't — those need the USB-serial-trainer route instead (drop the Arduino, point `rc_link` at the radio's serial port).

---

## Radio (EdgeTX) setup — one time
1. **Model → Trainer**: set trainer **input = PPM** (jack), mode so the *incoming* signal drives the channels. Assign a **trainer switch** to hand control to/from the laptop.
2. **Channel order** must equal `config/channels.yaml` + the `.ino` `failsafe[]` (default **AETR + arm**: roll, pitch, throttle, yaw, arm…). Fix it in the mixer if different.
3. Map the **arm** switch to the arm channel (ch5) so the laptop can arm/disarm.
4. Verify in **Model → Channels Monitor**: as the laptop sends, bars move on the right axes.

---

## Install + quick start
```bash
# Arduino: open arduino/ppm_trainer/ppm_trainer.ino in Arduino IDE, select board (Uno/Nano), upload.
# Laptop:
pip install pyserial pyyaml                 # for rc_link + keyboard
pip install opencv-python torch             # additionally for vision_deploy
ls /dev/tty*                                # find the Arduino port (ttyUSB0 / ttyACM0)

# packet self-test (no hardware):
python laptop/rc_link.py --selftest

# MODE 1 — manual (PROPS OFF!):
python laptop/keyboard_control.py --port /dev/ttyUSB0

# MODE 2 — policy, DRY RUN first (prints actions, no drone, no serial):
python laptop/vision_deploy.py --source rtsp://192.168.1.50:8554/cam --no-serial
# then PROPS OFF with the Arduino:
python laptop/vision_deploy.py --source rtsp://192.168.1.50:8554/cam \
    --port /dev/ttyUSB0 --hover-throttle <measured> --no-arm
```

**Keyboard controls:** `w/s` pitch · `a/d` roll · `q/e` yaw · `r/f` throttle · `space` centre · `t` arm · `g` disarm · `x` quit.

**How a policy action becomes sticks** (`policy_to_channels.py`):
`vx→pitch · vy→roll · yaw_rate→yaw · vz→throttle` (around `--hover-throttle`).

---

## Staged bring-up (IN ORDER — props OFF until Stage 3)
| Stage | What | Tool |
|---|---|---|
| **0** | bench, no drone: watch the radio's Channel Monitor; confirm axes + arm + that quitting → failsafe | `keyboard_control.py` |
| **1** | PROPS OFF, motors on drone: check motor directions/response match sticks; test **trainer-switch override** | `keyboard_control.py` |
| **2** | hover (tethered/open): manual control; **measure the hover throttle** (needed for the policy) | `keyboard_control.py` |
| **3** | scripted creep-forward (fixed channels, no perception) | `rc_link.py` directly |
| **4** | full autonomy | `vision_deploy.py` |

---

## Troubleshooting
| Symptom | Likely cause / fix |
|---|---|
| Radio channel bars don't move | trainer input not set to **PPM**, or trainer switch off; wrong jack pin (re-check Tip/Sleeve) |
| Bars move but **inverted/erratic** | try **inverted PPM** — flip `ON_STATE` (1↔0) in the `.ino` |
| Wrong axis moves (roll vs pitch) | channel order mismatch — align `channels.yaml`, `.ino` `failsafe[]`, and the radio mixer |
| Drone/motors keep cutting out | packets not streaming ≥2 Hz → failsafe; check `--port`, baud (115200), that the script is actually running |
| `could not open port` | wrong `--port`; `ls /dev/tty*`, and add yourself to the `dialout` group (`sudo usermod -aG dialout $USER`, re-login) |
| `cannot open video source` | RTSP URL wrong / stream not up; test with `ffplay rtsp://...` first, or use OBS Virtual Camera index |
| Everything works but it flies badly on policy | expected — that's the **sim2real gap**, not a wiring bug (see below) |

---

## Safety (non-negotiable)
- **Props off** for Stages 0–1. Open space, eye protection, everyone clear.
- The **radio trainer switch is your override** — flick it and the human pilot (sticks) takes over instantly. Always fly with a hand on it.
- The Arduino **failsafe** (throttle min + disarm on link loss) is a backstop, not a substitute for the trainer-switch override.
- Start every policy run with `--no-serial` (dry run), then `--no-arm`, before ever arming.

---

## Why an Arduino (and when you can skip it)
Plugging the radio straight into the laptop by USB makes it a **joystick input**
(radio→laptop) — the wrong direction. The trainer port is the reliable
“laptop→radio” path, and the Arduino turns serial into the PPM it expects. If
your radio supports **serial trainer over USB**, you can drop the Arduino and
point `rc_link` at the radio's serial port instead — same `rc_link`, different
device.
