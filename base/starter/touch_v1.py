#!/usr/bin/env python3
"""Touch v1: possession changes on touches, not on sustained proximity.

The ground truth comes from corrected tracking that names the team owning the ball, and the
challenge excludes touches without control and deflections from counting as a change. So:

  1. A touch is a sharp change in the ball's velocity with a player close enough to cause it. The
     velocity noise level is measured on the match itself (median + 5 robust standard deviations).
  2. A control is either a touch, or the ball staying with one team for K frames (K = 11, the
     turnover window read off the match in principled_v1).
  3. An opponent's touch only starts a candidate change. It is confirmed by that team's next touch
     or control, and cancelled if the team in possession touches the ball first (a deflection).
     A confirmed change is dated to the opponent's first touch.

Dead ball is taken unchanged from principled_v1 (p2): this model only decides which team has the
ball while it is alive.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FPS = 25
HOME, AWAY, NONE = 0, 1, -1

DEFAULT = {
    "half_window": 2,      # frames each side for the velocity estimate
    "dv_lag": 3,           # frames between the two velocities compared for a touch
    "noise_k": 5.0,        # touch threshold = median + k * 1.4826 * MAD of the velocity change
    "min_gap": 7,          # frames between two touches of the detector
    "radius": 1.5,         # metres from the ball for a player to be credited with a touch
    "dwell": 11,           # frames of exclusive closeness that count as control
    "use_dwell": True,
}


def ball_velocity(bx, by, h):
    """Central difference over +-h frames; NaN wherever an end point is missing."""
    n = len(bx)
    vx = np.full(n, np.nan)
    vy = np.full(n, np.nan)
    vx[h:n - h] = (bx[2 * h:] - bx[:n - 2 * h]) / (2 * h / FPS)
    vy[h:n - h] = (by[2 * h:] - by[:n - 2 * h]) / (2 * h / FPS)
    return vx, vy


def velocity_change(vx, vy, m):
    """|v(t+m) - v(t-m)|, the size of the kick the ball took around t."""
    n = len(vx)
    dv = np.full(n, np.nan)
    dv[m:n - m] = np.hypot(vx[2 * m:] - vx[:n - 2 * m], vy[2 * m:] - vy[:n - 2 * m])
    return dv


def robust_threshold(dv, k):
    x = dv[np.isfinite(dv)]
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return float(med + k * mad), float(med), float(mad)


def peaks(dv, thr, min_gap):
    """Local maxima above thr, greedily keeping the largest within min_gap frames."""
    cand = np.flatnonzero(np.nan_to_num(dv, nan=0.0) > thr)
    if not len(cand):
        return np.array([], dtype=int)
    order = cand[np.argsort(-dv[cand])]
    taken = np.zeros(len(dv), dtype=bool)
    keep = []
    for t in order:
        lo, hi = max(0, t - min_gap), min(len(dv), t + min_gap + 1)
        if taken[lo:hi].any():
            continue
        taken[t] = True
        keep.append(t)
    return np.sort(np.array(keep, dtype=int))


def team_distances(bx, by, px, py, team_of_track):
    """Per frame, the distance from the ball to the closest player of each team (NaN if none)."""
    out = []
    for team in (HOME, AWAY):
        cols = np.flatnonzero(team_of_track == team)
        d = np.hypot(px[:, cols] - bx[:, None], py[:, cols] - by[:, None])
        with np.errstate(invalid="ignore"):
            all_nan = np.all(np.isnan(d), axis=1)
            dmin = np.where(all_nan, np.nan, np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1))
        out.append(dmin)
    return out[0], out[1]


def detect(bx, by, d_home, d_away, cfg):
    """Touches as (frame, team) and the diagnostics of the detector."""
    vx, vy = ball_velocity(bx, by, cfg["half_window"])
    dv = velocity_change(vx, vy, cfg["dv_lag"])
    thr, med, mad = robust_threshold(dv, cfg["noise_k"])
    ts = peaks(dv, thr, cfg["min_gap"])
    touches = []
    for t in ts:
        dh, da = d_home[t], d_away[t]
        best = np.nanmin([dh, da]) if not (np.isnan(dh) and np.isnan(da)) else np.nan
        if np.isnan(best) or best > cfg["radius"]:
            continue
        touches.append((int(t), HOME if (not np.isnan(dh) and dh == best) else AWAY))
    info = {"dv_threshold": round(thr, 2), "dv_median": round(med, 2), "dv_mad": round(mad, 2),
            "peaks": int(len(ts)), "touches": len(touches)}
    return touches, ts, dv, info


def dwell_controls(d_home, d_away, radius, k):
    """(frame, team) at the start of every run of >= k frames where one team alone is closest."""
    events = []
    for team, d_own, d_other in ((HOME, d_home, d_away), (AWAY, d_away, d_home)):
        with np.errstate(invalid="ignore"):
            own = (d_own <= radius) & ~(d_other < d_own)
        edges = np.diff(np.concatenate(([0], own.view(np.int8), [0])))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if e - s >= k:
                events.append((int(s), team))
    return events


def assign(base_labels, touches, controls, graded):
    """Relabel the live stretches of base_labels from touches and controls.

    base_labels: array of 'H'/'A'/'D'/'X' (the dead-ball mask and the fallback team).
    Returns a new label array and the number of changes of possession it made.
    """
    labels = base_labels.copy()
    alive = graded & (base_labels != "D") & (base_labels != "X")
    edges = np.diff(np.concatenate(([0], alive.view(np.int8), [0])))
    stream = sorted([(t, team, "T") for t, team in touches] + [(t, team, "C") for t, team in controls])
    times = np.array([e[0] for e in stream], dtype=int)
    changes = 0
    for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        lo, hi = np.searchsorted(times, s), np.searchsorted(times, e)
        events = stream[lo:hi]
        if not events:
            continue                      # nothing seen: keep the base labels for this stretch
        cur, cur_start, pending = None, s, None
        spans = []
        for t, team, kind in events:
            if cur is None:
                cur, cur_start = team, s  # the first team to touch or control took the restart
                continue
            if team == cur:
                pending = None            # the team in possession touched it again: deflection
                continue
            if kind == "C" or (pending is not None and pending[1] == team):
                start = pending[0] if pending is not None else t
                spans.append((cur_start, start, cur))
                cur, cur_start, pending = team, start, None
                changes += 1
            else:
                pending = (t, team)
        spans.append((cur_start, e, cur))
        for a, b, team in spans:
            labels[a:b] = "H" if team == HOME else "A"
    return labels, changes


def predict(base_labels, ball_x, ball_y, px, py, team_of_track, graded, cfg=None):
    cfg = {**DEFAULT, **(cfg or {})}
    bx = ball_x.astype(np.float64)
    by = ball_y.astype(np.float64)
    d_home, d_away = team_distances(bx, by, px, py, team_of_track)
    touches, peaks_t, dv, info = detect(bx, by, d_home, d_away, cfg)
    controls = dwell_controls(d_home, d_away, cfg["radius"], cfg["dwell"]) if cfg["use_dwell"] else []
    labels, changes = assign(base_labels, touches, controls, graded)
    info.update({"controls": len(controls), "changes": changes})
    return labels, info, {"touches": touches, "peaks": peaks_t, "dv": dv,
                          "d_home": d_home, "d_away": d_away}
