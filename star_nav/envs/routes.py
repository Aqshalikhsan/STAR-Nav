"""Named long-corridor ROUTES for MockCorridorEnv -- hundreds of metres of winding
plantation corridor, instead of the original 50 m straight/zigzag strip.

DESIGN CONSTRAINT (why the map is a long strip, not a square):
The drone must always move FORWARD -- never double back. That means the centerline
is a function y = f(x) with x strictly increasing, so the flown arc length is
    L = integral sqrt(1 + (dy/dx)^2) dx   over the map's x-extent.
To get L = 700 m out of a 120 m x-extent you would need |dy/dx| ~ 5.7, i.e. the
drone flying ~80 deg off the forward axis, meandering almost perpendicular -- far
harder than the existing 40 deg zigzag that already took a 29-stage curriculum.
Lengthening the MAP instead (500 m x 120 m) gives the same 700 m at ~44 deg
average, which is in the learnable range. Hence: long strip, not a square.

A route is just the centerline y(x) plus its arc-length table. Progress along a
meandering route MUST be measured by ARC LENGTH, not by Euclidean distance to the
goal: on a winding path you can be near the goal in a straight line while still
having hundreds of metres of corridor left (and a meander can even take you
*away* from the goal, which would make a Euclidean progress reward punish correct
flying).

Existing worlds are untouched: this is opt-in via `env.route: <name>`; with no
route set MockCorridorEnv keeps its original straight/zigzag behaviour exactly.
"""
from __future__ import annotations

import numpy as np

# name -> (length_x_m, width_y_m, builder)
_REGISTRY = {}


def register(name):
    def deco(fn):
        _REGISTRY[name] = fn
        return fn
    return deco


class Route:
    """Centerline y(x) sampled densely, with a cumulative arc-length table."""

    def __init__(self, x, y, length_x, width_y, name=""):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.length_x = float(length_x)
        self.width_y = float(width_y)
        self.name = name
        d = np.hypot(np.diff(self.x), np.diff(self.y))
        self.s = np.concatenate([[0.0], np.cumsum(d)]).astype(np.float32)  # arc length
        self.total_s = float(self.s[-1])

    # --- centerline queries (x is monotonic, so everything is a function of x) ---
    def center_y(self, x):
        return np.interp(np.asarray(x, dtype=np.float32), self.x, self.y).astype(np.float32)

    def arclen(self, x):
        """Arc length travelled along the corridor by the time you reach this x."""
        return np.interp(np.asarray(x, dtype=np.float32), self.x, self.s).astype(np.float32)

    def tangent(self, x):
        """Corridor heading (rad) at x -- the 'forward' the drone should align with."""
        xa = np.asarray(x, dtype=np.float32)
        i = np.clip(np.searchsorted(self.x, xa, side="right") - 1, 0, len(self.x) - 2)
        dx = self.x[i + 1] - self.x[i]
        dy = self.y[i + 1] - self.y[i]
        ang = np.arctan2(dy, np.maximum(dx, 1e-6))
        return float(ang) if np.ndim(x) == 0 else ang.astype(np.float32)

    def max_turn_deg(self):
        t = np.degrees(self.tangent(self.x[:-1]))
        return float(np.abs(t).max())

    def summary(self):
        return (f"route '{self.name}': map {self.length_x:.0f} x {self.width_y:.0f} m, "
                f"path {self.total_s:.0f} m ({self.total_s / self.length_x:.2f}x the x-extent), "
                f"max heading {self.max_turn_deg():.0f} deg off-axis")


# ----------------------------------------------------------------------------
# Segment primitives -- each returns y over its own x span. All are functions of
# x, which is what guarantees the drone never doubles back.
# ----------------------------------------------------------------------------
def _straight(xs, y0):
    return np.full_like(xs, y0)


def _sine(xs, y0, amp, wavelength, phase=0.0):
    return y0 + amp * np.sin(2 * np.pi * (xs - xs[0]) / wavelength + phase)


def _dogleg(xs, y0, amp, n_legs):
    """Piecewise-linear zigzag: sharp corners, like the existing zigzag corridor."""
    anchors_x = np.linspace(xs[0], xs[-1], n_legs + 1)
    anchors_y = y0 + amp * np.array([(-1.0) ** k for k in range(n_legs + 1)], dtype=np.float32)
    anchors_y[0] = y0
    anchors_y[-1] = y0
    return np.interp(xs, anchors_x, anchors_y)


def _sweep(xs, y0, amp):
    """One WIDE, smooth lateral shift -- a long sweeping bend, the gentlest segment.
    Peak slope is amp*pi/(2*span), so a long span keeps it shallow. (The first
    version multiplied a full sine by a window and peaked at 80 deg -- steeper than
    the "hard" meander it was supposed to relieve.)"""
    t = (xs - xs[0]) / (xs[-1] - xs[0])
    return y0 + amp * (1.0 - np.cos(np.pi * t)) / 2.0


@register("plantation_long")
def plantation_long(rng=None, ds=0.5):
    """ONE long world with VARIED corridor character along it (the user's ask:
    "1 world, tapi di dalamnya macem-macem"). ~700 m of corridor across a
    500 x 120 m plantation, always progressing forward.

    Difficulty deliberately ramps and varies so a single episode exercises
    everything: easy on-ramp -> gentle S -> sharp doglegs -> tight meander (the
    hard part) -> wide sweep -> doglegs again -> gentle -> straight run-out.
    """
    rng = rng or np.random.default_rng(0)
    LX, LY = 500.0, 120.0
    y0 = LY / 2.0
    # (x_end, kind, params) -- built left to right
    # Amplitudes/wavelengths are tuned so the STEEPEST heading stays ~55-60 deg off
    # the forward axis -- comparable to the existing 40 deg zigzag, i.e. inside the
    # learnable range. (A first cut at amp=38/lambda=50 gave 78-82 deg and a 953 m
    # path: too steep to learn, see the module docstring.) For a sine the peak
    # slope is amp*2*pi/wavelength; for a dogleg it is 2*amp/leg_length.
    plan = [
        (50, "straight", {}),                                # easy on-ramp
        (130, "sine", dict(amp=16, wavelength=95)),          # gentle S        (~46 deg)
        (200, "dogleg", dict(amp=17, n_legs=3)),             # sharp corners   (~54 deg)
        (310, "sine", dict(amp=24, wavelength=100)),         # tight meander   (~56 deg, hardest)
        (380, "sweep", dict(amp=-30)),                       # wide sweeping bend (~34 deg)
        (440, "dogleg", dict(amp=15, n_legs=2)),             # sharp again     (~45 deg)
        (480, "sine", dict(amp=14, wavelength=80)),          # gentle
        (500, "straight", {}),                               # run-out to the finish
    ]
    xs_all, ys_all, x0 = [], [], 0.0
    for x_end, kind, kw in plan:
        xs = np.arange(x0, x_end, ds, dtype=np.float32)
        if len(xs) < 2:
            x0 = x_end
            continue
        y_start = ys_all[-1][-1] if ys_all else y0
        if kind == "straight":
            ys = _straight(xs, y_start)
        elif kind == "sine":
            ys = _sine(xs, y_start, **kw)
        elif kind == "dogleg":
            ys = _dogleg(xs, y_start, **kw)
        elif kind == "sweep":
            ys = _sweep(xs, y_start, **kw)
        else:
            raise ValueError(kind)
        xs_all.append(xs); ys_all.append(ys.astype(np.float32))
        x0 = x_end
    x = np.concatenate(xs_all)
    y = np.clip(np.concatenate(ys_all), 8.0, LY - 8.0)       # keep inside the plantation
    return Route(x, y, LX, LY, name="plantation_long")


def make_route(name, rng=None):
    if name not in _REGISTRY:
        raise KeyError(f"unknown route '{name}'; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](rng=rng)


def route_names():
    return sorted(_REGISTRY)
