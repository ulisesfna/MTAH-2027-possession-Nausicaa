#!/usr/bin/env python3
"""Principled v1: markius_v1 with every tuned constant replaced by a definition or a value
derived from the match itself, without ground truth.

The decision logic is markius_v1's state machine, rewritten with switches so that each change can
be measured on its own. CONFIGS["v1"] reproduces markius_v1 exactly; CONFIGS["p3"] is the model:
p2 plus turnovers dated to the middle of the ball's flight (starter/timing_v1.py). p1 and p2_ki are
kept as measured, rejected alternatives.
Kinematics extraction is copied unchanged from markius_v1 (Marc's code), so both share the cache.

    python models/principled_v1.py                  # p3, writes evaluation/results/principled_v1.csv
    python models/principled_v1.py --config v1      # must match markius_v1.csv
"""

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import games  # noqa: E402

HOME, AWAY, NOBODY = 0, 1, -1
FPS = 25

def get_team_mapping(meta_path):
    """Map tracking slots to teams (HOME=0, AWAY=1) from EPTS metadata."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    slots = [p.get("teamId") for p in root.iter("Player")]
    return np.array([HOME if t_id == home_id else AWAY for t_id in slots], dtype=np.int8)


def compute_velocity_1d(x, dt=1.0 / 25.0):
    """Compute central difference velocity with boundary forward/backward fallbacks."""
    n = len(x)
    v = np.full(n, np.nan, dtype=np.float32)
    dx_f = np.diff(x) / dt
    f_valid = ~np.isnan(dx_f)
    v[:-1][f_valid] = dx_f[f_valid]
    dx_b = np.diff(x) / dt
    b_valid = ~np.isnan(dx_b)
    v[1:][b_valid] = (v[1:][b_valid] + dx_b[b_valid]) * 0.5 if np.any(f_valid) else dx_b[b_valid]
    dx_c = (x[2:] - x[:-2]) / (2.0 * dt)
    c_valid = ~np.isnan(dx_c)
    v[1:-1][c_valid] = dx_c[c_valid]
    return v


def filter_min_duration(mask, min_len):
    """Filter out active intervals shorter than min_len frames."""
    out = mask.copy()
    diff = np.diff(np.concatenate(([0], out.view(np.int8), [0])))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    for s, e in zip(starts, ends):
        if e - s < min_len:
            out[s:e] = False
    return out


def extract_or_load_kinematics(raw_path, meta_path, n_frames, pitch, cache_path=None):
    """Extract or load player and ball kinematics with autonomous glitch filtering."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading kinematics from cache: {cache_path}")
        data = np.load(cache_path)
        return {
            "ball_x_raw": data["ball_x_raw"],
            "ball_y_raw": data["ball_y_raw"],
            "ball_x_recon": data["ball_x_recon"],
            "ball_y_recon": data["ball_y_recon"],
            "ball_is_imputed": data["ball_is_imputed"],
            "ball_gap_len": data["ball_gap_len"],
            "mean_speed": data["mean_speed"],
            "home_min_dist": data["home_min_dist"],
            "away_min_dist": data["away_min_dist"],
            "home_min_kappa": data["home_min_kappa"],
            "away_min_kappa": data["away_min_kappa"],
            "home_best_player": data["home_best_player"],
            "away_best_player": data["away_best_player"],
            "home_rel_speed": data["home_rel_speed"],
            "away_rel_speed": data["away_rel_speed"],
            "team_of_track": data["team_of_track"],
        }

    team_of_track = get_team_mapping(meta_path)
    px, py, mean_speed, ball_x_raw, ball_y_raw = parse_atd(raw_path, n_frames, len(team_of_track), pitch)
    out = kinematics_from_arrays(px, py, mean_speed, ball_x_raw, ball_y_raw, team_of_track, pitch)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            **{k: out[k] for k in (
                "ball_x_raw", "ball_y_raw", "ball_x_recon", "ball_y_recon", "ball_is_imputed",
                "ball_gap_len", "mean_speed", "home_min_dist", "away_min_dist", "home_min_kappa",
                "away_min_kappa", "home_best_player", "away_best_player", "home_rel_speed",
                "away_rel_speed", "team_of_track")},
        )

    return out


def parse_atd(raw_path, n_frames, n_tracks, pitch):
    """Raw ATD into arrays: slot positions, mean player speed and ball, in metres from the corner."""
    length, width = pitch

    px = np.full((n_frames, n_tracks), np.nan, dtype=np.float32)
    py = np.full((n_frames, n_tracks), np.nan, dtype=np.float32)
    mean_speed = np.zeros(n_frames, dtype=np.float32)
    ball_x_raw = np.full(n_frames, np.nan, dtype=np.float32)
    ball_y_raw = np.full(n_frames, np.nan, dtype=np.float32)

    print(f"Parsing raw tracking data from {raw_path}...")
    with open(raw_path, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            vf = int(parts[0]) - 1
            if vf >= n_frames:
                break
            speeds = []
            for slot_idx, group in enumerate(parts[1].split(";")):
                if group[0] == "N":
                    continue
                c1 = group.index(",")
                c2 = group.index(",", c1 + 1)
                px[vf, slot_idx] = float(group[:c1]) * length
                py[vf, slot_idx] = float(group[c1 + 1:c2]) * width
                sp_str = group[c2 + 1:]
                if sp_str[0] != "N":
                    speeds.append(float(sp_str))

            if speeds:
                mean_speed[vf] = np.mean(speeds)

            ball = parts[2]
            if ball[0] != "N":
                c1 = ball.index(",")
                c2 = ball.index(",", c1 + 1)
                ball_x_raw[vf] = float(ball[:c1]) * length
                ball_y_raw[vf] = float(ball[c1 + 1:c2]) * width

    return px, py, mean_speed, ball_x_raw, ball_y_raw


def kinematics_from_arrays(px, py, mean_speed, ball_x_raw, ball_y_raw, team_of_track, pitch):
    """Kinematics from position arrays, whatever loaded them (the ATD here, bench/adapter.py for
    the external matches). px, py: (n_frames, n_tracks) in metres from the top-left corner."""
    length, width = pitch
    n_frames, n_tracks = px.shape
    # 1. Non-physical Transport Velocity Glitch Filter (> 12 m/s jump)
    v_max_transport = 12.0
    for slot in range(n_tracks):
        valid_idx = np.flatnonzero(~np.isnan(px[:, slot]))
        if len(valid_idx) < 2:
            continue
        diff_idx = np.diff(valid_idx)
        for i, gap in enumerate(diff_idx):
            if gap > 1:
                f_prev = valid_idx[i]
                f_curr = valid_idx[i + 1]
                dt = (f_curr - f_prev) / 25.0
                d_jump = np.hypot(px[f_curr, slot] - px[f_prev, slot], py[f_curr, slot] - py[f_prev, slot])
                if (d_jump / dt) > v_max_transport:
                    end_burst = f_curr
                    while end_burst < n_frames - 1 and not np.isnan(px[end_burst + 1, slot]):
                        end_burst += 1
                    px[f_curr:end_burst + 1, slot] = np.nan
                    py[f_curr:end_burst + 1, slot] = np.nan

    # 2. Piecewise linear interpolation for short ball gaps (<= 25 frames / 1.0s)
    bx_recon = ball_x_raw.copy()
    by_recon = ball_y_raw.copy()
    ball_is_imputed = np.zeros(n_frames, dtype=bool)
    ball_gap_len = np.zeros(n_frames, dtype=np.int16)

    nan_b = np.isnan(bx_recon)
    diff = np.diff(np.concatenate(([0], nan_b.view(np.int8), [0])))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    for s, e in zip(starts, ends):
        gap_len = e - s
        if gap_len <= 25 and s > 0 and e < n_frames:
            if not np.isnan(bx_recon[s - 1]) and not np.isnan(bx_recon[e]):
                if (-1.0 <= bx_recon[s - 1] <= length + 1.0) and (-1.0 <= bx_recon[e] <= length + 1.0):
                    bx_recon[s:e] = np.linspace(bx_recon[s - 1], bx_recon[e], gap_len + 2)[1:-1]
                    by_recon[s:e] = np.linspace(by_recon[s - 1], by_recon[e], gap_len + 2)[1:-1]
                    ball_is_imputed[s:e] = True
                    ball_gap_len[s:e] = gap_len

    v_bx = compute_velocity_1d(bx_recon)
    v_by = compute_velocity_1d(by_recon)

    features = {}
    r0 = 0.85
    tau_stride = 0.25
    max_reach_spd = 8.5

    for team_code, name in [(HOME, "home"), (AWAY, "away")]:
        slots = np.where(team_of_track == team_code)[0]
        t_px = px[:, slots]
        t_py = py[:, slots]
        tracks = slots.astype(np.int16)

        dx = bx_recon[:, None] - t_px
        dy = by_recon[:, None] - t_py
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 0.05)
        u_x = dx / dist_safe
        u_y = dy / dist_safe

        vx = np.gradient(t_px, axis=0) * 25.0
        vy = np.gradient(t_py, axis=0) * 25.0

        v_app = vx * u_x + vy * u_y
        v_app_fwd = np.clip(np.nan_to_num(v_app, nan=0.0), 0.0, max_reach_spd)
        reach = r0 + tau_stride * v_app_fwd
        kappa = dist / np.maximum(reach, 0.1)

        sub_k_clean = np.where(np.isnan(kappa), 1e9, kappa)
        best_slot = np.argmin(sub_k_clean, axis=1)
        min_kappa = np.take_along_axis(kappa, best_slot[:, None], axis=1).squeeze(1)
        min_dist = np.take_along_axis(dist, best_slot[:, None], axis=1).squeeze(1)
        best_player = tracks[best_slot]

        best_vx = np.take_along_axis(vx, best_slot[:, None], axis=1).squeeze(1)
        best_vy = np.take_along_axis(vy, best_slot[:, None], axis=1).squeeze(1)
        rel_vx = best_vx - v_bx
        rel_vy = best_vy - v_by
        rel_speed = np.sqrt(rel_vx * rel_vx + rel_vy * rel_vy)

        all_nan = np.all(np.isnan(kappa), axis=1)
        best_player[all_nan] = -1
        min_kappa[all_nan] = 1e9
        min_dist[all_nan] = np.nan
        rel_speed[all_nan] = np.nan

        features[name] = {
            "min_dist": min_dist,
            "min_kappa": min_kappa,
            "best_player": best_player.astype(np.int16),
            "rel_speed": rel_speed.astype(np.float32),
        }

    return {
        "ball_x_raw": ball_x_raw,
        "ball_y_raw": ball_y_raw,
        "ball_x_recon": bx_recon,
        "ball_y_recon": by_recon,
        "ball_is_imputed": ball_is_imputed,
        "ball_gap_len": ball_gap_len,
        "mean_speed": mean_speed,
        "home_min_dist": features["home"]["min_dist"],
        "away_min_dist": features["away"]["min_dist"],
        "home_min_kappa": features["home"]["min_kappa"],
        "away_min_kappa": features["away"]["min_kappa"],
        "home_best_player": features["home"]["best_player"],
        "away_best_player": features["away"]["best_player"],
        "home_rel_speed": features["home"]["rel_speed"],
        "away_rel_speed": features["away"]["rel_speed"],
        "team_of_track": team_of_track,
    }


def load_smart_tagging_phases(st_path, n_frames):
    tree = ET.parse(st_path)
    st_home = np.zeros(n_frames, dtype=bool)
    st_away = np.zeros(n_frames, dtype=bool)
    st_set_pieces = np.zeros(n_frames, dtype=bool)

    offensive_codes = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL"}

    for inst in tree.getroot().iter("instance"):
        s = max(0, min(n_frames - 1, round(float(inst.find("start").text) * 25)))
        e = max(0, min(n_frames, round(float(inst.find("end").text) * 25)))
        c = inst.find("code").text
        team = None
        for l in inst.iter("label"):
            if l.find("group") is not None and l.find("text") is not None:
                if l.find("group").text == "Team":
                    team = l.find("text").text
        if c in offensive_codes:
            if team == "Barcelona":
                st_home[s:e] = True
            elif team == "Atletic Club":
                st_away[s:e] = True
        elif c == "SET PIECES":
            st_set_pieces[s:e] = True

    return st_home, st_away, st_set_pieces

# -- configurations ----------------------------------------------------------------------------
#
# "v1" reproduces markius_v1 exactly, including its constants. "p1" is the principled model. Every
# other entry is v1 with exactly one change, so the effect of each change can be read on its own.

V1 = {
    "kappa": 0.95,              # control if distance / reach <= kappa
    "penalty": True,            # down-weight control on interpolated ball frames
    "confirm": True,            # turnover needs "same player" or "moves with the ball"
    "backdate": "min_rel_speed",
    "turnover_frames": 11,      # consecutive frames of opponent control for a turnover
    "dead_speed": (1.40, 1.90),  # (ball visible, ball missing) player-speed thresholds, m/s
    "visible_dead": "st_veto",  # how a stoppage with the ball visible is recognised
    "init": "v1",               # frames of control needed to start a possession after a stoppage
    "oob_margin": 0.2,          # metres past the line before the ball counts as out
    "fix_halves": False,        # label the last frame of each half
}

P1 = {
    "kappa": 1.0,               # definition of reach: the player can get to the ball
    "penalty": False,           # measured inert on the full match
    "confirm": False,           # measured inert on the full match
    "backdate": "run_start",    # possession changes when the opponent's control begins
    "turnover_frames": "otsu",  # split brief touches from sustained control, on this match
    "dead_speed": "otsu",       # split dead from live player speed, on this match
    "visible_dead": "ball_still",  # Smart Tagging phases run over stoppages, so not used here
    "init": "first",            # after a stoppage there is always a clear possessor
    "oob_margin": 0.11,         # ball radius: out once the whole ball is past the line
    "fix_halves": True,
}

# p2: v1 with the inert parts removed, the half-end bug fixed, and the two thresholds that can be
# read off the match without labels. Decided before running it (31/10/2026): the dead-ball speed
# from Kittler-Illingworth is kept only if it puts 40-52% of graded frames in dead ball; otherwise
# p2 keeps v1's 1.40 / 1.90 m/s and the document states them as chosen values.
#
# Result on the full match (31/10/2026): Kittler-Illingworth gave 4.05 m/s (fitted means 1.68 and
# 4.75), which puts 67% of graded frames in dead ball. Outside the range, so p2 keeps 1.40 / 1.90.
# p2 reproduces markius_v1's output exactly, with the inert code removed, the half-end frames
# fixed, and the 11-frame turnover window now read off the match instead of chosen.
P2_KI = {**V1, "penalty": False, "confirm": False, "fix_halves": True,
         "turnover_frames": "otsu", "dead_speed": "ki"}
P2 = {**P2_KI, "dead_speed": V1["dead_speed"]}

# Judgement values kept from v1, stated here so the technical document can list them.
SPEED_WINDOW = 60       # frames over which player speed is averaged (2.4 s)
MIN_DEAD = 90           # shortest stoppage accepted (3.6 s)
MIN_OUT = 15            # frames the ball must stay outside before it counts as out (0.6 s)
MAX_TRANSIT = 2         # frames of no control tolerated inside an opponent's control run
BALL_STILL = 1.0        # m/s; a ball slower than walking pace over the window is not in play
CONTROL_REL_SPEED = 3.5  # m/s; markius_v1's "moves with the player" bound, used by confirm "rel_speed"

# p3: p2 with every direct turnover moved to the middle of the ball's flight between the last
# contact of the team losing it and the first contact of the team winning it. The public ground truth
# dates turnovers that way; p2 dated them on arrival, 2-3 s late. Measured 31/10/2026: on the public
# 5 minutes the median turnover start error goes from 2.30 s to 1.02 s (spellF1 0.588 -> 0.612,
# macroF1 0.886 -> 0.897). The rule was read off those same 5 minutes, so that is not an independent
# test; it carries over to the full match only if the labelling convention does.
P3 = {**P2, "retime": "midpoint"}

# p4 candidates, both taken from the definition in CHALLENGE.md, no new constant:
#   contest "keep": when both teams are within reach, possession stays with the team that had it;
#   confirm "rel_speed": a turnover needs the ball to move with the new player (median relative
#   speed <= 3.5 m/s, markius_v1's own value, inert there), so a header or a deflection is not one.
P4_CONTEST = {**P3, "contest": "keep"}
P4_CONTROL = {**P3, "confirm": "rel_speed"}
P4_BOTH = {**P3, "contest": "keep", "confirm": "rel_speed"}

# p3 without the Smart Tagging, for the external matches (which have none): the visible-ball
# stoppage and the restart rule take P1's alternatives, no new constant.
P3_NOTAG = {**P3, "visible_dead": P1["visible_dead"], "init": P1["init"]}

# Variants of p3_notag for the external bench (16/11/2026). p4_contest and p4_control were dropped on
# the public 5 minutes only, which cannot decide anything, so they are measured again here, plus:
#   restart "backdate": after a stoppage the first possession starts where play restarts, not at the
#   first frame of detected control ("whenever play restarts, there is a clear possessor"). Without
#   it the frames in between stay NOBODY and come out as D. No new constant.
#   p2_notag: p3_notag without the mid-flight retime, informative only (never validated outside the
#   public window).
# control_rel_speed 3.0 and 4.0 are only a sensitivity check of the inherited 3.5 m/s.
NOTAG_VARIANTS = {
    "p3_notag_contest": {**P3_NOTAG, "contest": "keep"},
    "p3_notag_control": {**P3_NOTAG, "confirm": "rel_speed"},
    "p3_notag_control30": {**P3_NOTAG, "confirm": "rel_speed", "control_rel_speed": 3.0},
    "p3_notag_control40": {**P3_NOTAG, "confirm": "rel_speed", "control_rel_speed": 4.0},
    "p3_notag_restart": {**P3_NOTAG, "restart": "backdate"},
    "p2_notag": {**P3_NOTAG, "retime": None},
}

CONFIGS = {"v1": V1, "p1": P1, "p2": P2, "p2_ki": P2_KI, "p3": P3, "p3_notag": P3_NOTAG,
           "p4_contest": P4_CONTEST, "p4_control": P4_CONTROL, "p4_both": P4_BOTH, **NOTAG_VARIANTS}


def notag_combo(names):
    """p3_notag plus the changes of several variants, e.g. ("contest", "restart")."""
    cfg = dict(P3_NOTAG)
    for n in names:
        extra = {k: v for k, v in NOTAG_VARIANTS[f"p3_notag_{n}"].items() if P3_NOTAG.get(k) != v}
        cfg.update(extra)
    return cfg


def get_config(name):
    """CONFIGS entry, or a combination written as p3_notag+contest+restart."""
    if name in CONFIGS:
        return CONFIGS[name]
    if name.startswith("p3_notag+"):
        return notag_combo(name.split("+")[1:])
    raise KeyError(name)
for key in P1:
    if P1[key] != V1[key]:
        CONFIGS[f"v1+{key}"] = {**V1, key: P1[key]}
CONFIGS["v1+dead_speed_ki"] = {**V1, "dead_speed": "ki"}


# -- thresholds read off the match --------------------------------------------------------------

def otsu(values, bins=256):
    """Otsu's threshold for a 1-D sample, and eta = between-class / total variance in [0, 1].

    eta near 1 means two well separated groups; below roughly 0.6 the split is not meaningful.
    """
    x = values[np.isfinite(values)]
    hist, edges = np.histogram(x, bins=bins)
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    mu0 = np.cumsum(p * centers)
    mu_t = mu0[-1]
    w1 = 1 - w0
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w0 - mu0) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = 0
    k = int(np.argmax(between))
    eta = between[k] / x.var() if x.var() > 0 else 0.0
    return float(edges[k + 1]), float(eta)


def kittler_illingworth(values, bins=256):
    """Minimum-error threshold: the cut that best fits two Gaussians of unequal spread and size.

    Otsu assumes both groups have similar spread, which the player-speed distribution does not.
    Returns the threshold and the two fitted means, so a split with no real second group shows.
    """
    x = values[np.isfinite(values)]
    hist, edges = np.histogram(x, bins=bins)
    p = hist / hist.sum()
    c = (edges[:-1] + edges[1:]) / 2
    best, best_k, means = np.inf, None, None
    for k in range(1, bins - 1):
        p1, p2 = p[:k + 1].sum(), p[k + 1:].sum()
        if p1 < 1e-3 or p2 < 1e-3:
            continue
        m1 = (p[:k + 1] * c[:k + 1]).sum() / p1
        m2 = (p[k + 1:] * c[k + 1:]).sum() / p2
        v1 = (p[:k + 1] * (c[:k + 1] - m1) ** 2).sum() / p1
        v2 = (p[k + 1:] * (c[k + 1:] - m2) ** 2).sum() / p2
        if v1 <= 0 or v2 <= 0:
            continue
        j = 1 + p1 * np.log(v1) + p2 * np.log(v2) - 2 * (p1 * np.log(p1) + p2 * np.log(p2))
        if j < best:
            best, best_k, means = j, k, (m1, m2)
    return float(edges[best_k + 1]), means


def run_lengths(mask):
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def ball_speed(bx, by):
    v_bx = np.diff(bx, prepend=bx[0]) * 25.0
    v_by = np.diff(by, prepend=by[0]) * 25.0
    return np.hypot(v_bx, v_by)


def smooth(x, window):
    """Moving mean that ignores NaN; frames with no valid sample in the window stay NaN."""
    valid = np.isfinite(x)
    k = np.ones(window)
    s = np.convolve(np.where(valid, x, 0.0), k, mode="same")
    c = np.convolve(valid.astype(float), k, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


# -- the model ---------------------------------------------------------------------------------

def predict(kinematics, st, n_frames, half_blocks, kickoff_2_frame, pitch, cfg):
    """Labels for every frame, and a dict of the thresholds that were used."""
    bx = kinematics["ball_x_recon"]
    by = kinematics["ball_y_recon"]
    mean_speed = kinematics["mean_speed"]
    # Same dtypes and operation order as markius_v1, so CONFIGS["v1"] reproduces it bit for bit.
    h_eff = kinematics["home_min_kappa"]
    a_eff = kinematics["away_min_kappa"]
    home_best, away_best = kinematics["home_best_player"], kinematics["away_best_player"]
    home_rel, away_rel = kinematics["home_rel_speed"], kinematics["away_rel_speed"]
    length, width = pitch
    info = {}

    half1_start, half1_end = half_blocks[0][1], half_blocks[0][2]
    half2_start, half2_end = half_blocks[1][1], half_blocks[1][2]
    graded = np.zeros(n_frames, dtype=bool)
    graded[half1_start:half1_end + 1] = True
    graded[half2_start:half2_end + 1] = True

    if cfg["penalty"]:
        w_gap = np.where(kinematics["ball_is_imputed"],
                         1.0 - 0.4 * (kinematics["ball_gap_len"] / 25.0), 1.0).astype(np.float32)
        h_eff = kinematics["home_min_kappa"] / w_gap
        a_eff = kinematics["away_min_kappa"] / w_gap

    # Dead ball
    is_dead = np.zeros(n_frames, dtype=bool)
    is_dead[half2_start:kickoff_2_frame] = True

    m = cfg["oob_margin"]
    outside = (~np.isnan(bx)) & ((bx < -m) | (bx > length + m) | (by < -m) | (by > width + m))
    is_dead |= filter_min_duration(outside, MIN_OUT)

    smooth_speed = np.convolve(mean_speed, np.ones(SPEED_WINDOW) / SPEED_WINDOW, mode="same")
    if cfg["dead_speed"] == "otsu":
        thr, eta = otsu(smooth_speed[graded & (smooth_speed > 0)])
        visible_thr = missing_thr = thr
        info["dead_speed"] = (round(thr, 3), round(eta, 3))
    elif cfg["dead_speed"] == "ki":
        thr, (m1, m2) = kittler_illingworth(smooth_speed[graded & (smooth_speed > 0)])
        visible_thr = missing_thr = thr
        info["dead_speed"] = (round(thr, 3), f"means {m1:.2f}/{m2:.2f}")
    else:
        visible_thr, missing_thr = cfg["dead_speed"]

    missing_stoppage = np.isnan(bx) & (smooth_speed < missing_thr)
    is_dead |= filter_min_duration(missing_stoppage, MIN_DEAD)

    slow = (smooth_speed < visible_thr) & (smooth_speed > 0)
    if cfg["visible_dead"] == "st_veto":
        slow &= ~(st["home"] | st["away"])
    else:
        slow &= smooth(ball_speed(bx, by), SPEED_WINDOW) < BALL_STILL
    is_dead |= filter_min_duration(slow, MIN_DEAD)

    v_ball = ball_speed(bx, by)

    restart_zone = np.zeros(n_frames, dtype=bool)
    if cfg["init"] == "v1":
        dead_diff = np.diff(np.concatenate(([0], is_dead.view(np.int8), [0])))
        for de in np.flatnonzero(dead_diff == -1):
            restart_zone[de:min(n_frames, de + 15)] = True
        frames = np.arange(n_frames)
        restart_zone |= np.abs(frames - half1_start) <= 30
        restart_zone |= np.abs(frames - kickoff_2_frame) <= 30
        restart_zone |= st["set_pieces"]

    kappa = cfg["kappa"]
    h_ctrl = (~np.isnan(h_eff)) & (h_eff <= kappa) & (~is_dead)
    a_ctrl = (~np.isnan(a_eff)) & (a_eff <= kappa) & (~is_dead)

    if cfg["turnover_frames"] == "otsu":
        excl_h = h_ctrl & (~a_ctrl | (h_eff < a_eff - 1e-4))
        excl_a = a_ctrl & (~h_ctrl | (a_eff < h_eff - 1e-4))
        lengths = np.concatenate((run_lengths(excl_h), run_lengths(excl_a))).astype(float)
        thr, eta = otsu(np.log(lengths))
        K = max(2, int(math.ceil(math.exp(thr))))
        info["turnover_frames"] = (K, round(eta, 3), len(lengths))
    else:
        K = cfg["turnover_frames"]

    state_seq = np.full(n_frames, NOBODY, dtype=np.int8)
    inst_seq = np.full(n_frames, NOBODY, dtype=np.int8)
    curr_state = NOBODY
    opp_cand = NOBODY
    opp_frames = 0
    opp_start_frame = -1
    transit_frames = 0
    cand_history = deque(maxlen=K)
    live_start = 0  # first frame after the last stoppage (restart "backdate")

    for f in range(n_frames):
        if is_dead[f]:
            live_start = f + 1
            curr_state = NOBODY
            opp_cand = NOBODY
            opp_frames = 0
            opp_start_frame = -1
            transit_frames = 0
            cand_history.clear()
            state_seq[f] = NOBODY
            continue

        hc, ac = h_ctrl[f], a_ctrl[f]
        if hc and not ac:
            inst = HOME
        elif ac and not hc:
            inst = AWAY
        elif hc and ac and cfg.get("contest") == "keep" and curr_state != NOBODY:
            inst = curr_state        # contested ball: possession stays with the last team (CHALLENGE.md)
        elif hc and ac:
            if h_eff[f] < a_eff[f] - 1e-4:
                inst = HOME
            elif a_eff[f] < h_eff[f] - 1e-4:
                inst = AWAY
            else:
                inst = curr_state if curr_state != NOBODY else NOBODY
        else:
            inst = NOBODY
        inst_seq[f] = inst

        if curr_state == NOBODY:
            if inst != NOBODY:
                if opp_cand == inst:
                    opp_frames += 1
                else:
                    opp_cand = inst
                    opp_frames = 1
                    opp_start_frame = f
                if cfg["init"] == "v1":
                    required = 1 if (restart_zone[f] or v_ball[f] >= 2.0) else 3
                else:
                    required = 1
                if opp_frames >= required:
                    curr_state = opp_cand
                    first = live_start if cfg.get("restart") == "backdate" else opp_start_frame
                    state_seq[first:f + 1] = curr_state
                    opp_cand = NOBODY
                    opp_frames = 0
            else:
                opp_cand = NOBODY
                opp_frames = 0
            state_seq[f] = curr_state
            continue

        opp = AWAY if curr_state == HOME else HOME
        if inst == opp:
            best_p = away_best[f] if opp == AWAY else home_best[f]
            rel_s = away_rel[f] if opp == AWAY else home_rel[f]
            if opp_cand == opp:
                opp_frames += 1
            else:
                opp_cand = opp
                opp_frames = 1
                opp_start_frame = f
                cand_history.clear()
            cand_history.append((f, best_p, rel_s))
            transit_frames = 0

            if len(cand_history) == K:
                valid_pairs = [(p[0], p[2]) for p in cand_history if not np.isnan(p[2])]
                if cfg["confirm"] == "rel_speed":
                    # control: the ball moves with the player, not past him (a header, a deflection)
                    lim = cfg.get("control_rel_speed", CONTROL_REL_SPEED)
                    accept = (np.median([p[1] for p in valid_pairs]) <= lim) if valid_pairs else False
                elif cfg["confirm"]:
                    players = [p[1] for p in cand_history if p[1] >= 0]
                    if players:
                        _, counts = np.unique(players, return_counts=True)
                        same_player = np.max(counts) / len(players) >= 0.50
                    else:
                        same_player = False
                    moves = (np.median([p[1] for p in valid_pairs]) <= 3.5) if valid_pairs else False
                    accept = same_player or moves
                else:
                    accept = True
                if accept:
                    curr_state = opp
                    if cfg["backdate"] == "min_rel_speed" and valid_pairs:
                        start = min(valid_pairs, key=lambda p: p[1])[0]
                    else:
                        start = cand_history[0][0]
                    state_seq[start:f + 1] = curr_state
                    opp_cand = NOBODY
                    opp_frames = 0
                    cand_history.clear()
                else:
                    state_seq[f] = curr_state
            else:
                state_seq[f] = curr_state
        elif inst == NOBODY:
            if opp_cand != NOBODY:
                transit_frames += 1
                if transit_frames > MAX_TRANSIT:
                    opp_cand = NOBODY
                    opp_frames = 0
                    transit_frames = 0
                    cand_history.clear()
            state_seq[f] = curr_state
        else:
            opp_cand = NOBODY
            opp_frames = 0
            transit_frames = 0
            cand_history.clear()
            state_seq[f] = curr_state

    if cfg.get("exposa_control"):
        # control clar: pilota morta per regla, o l'equip que controla ara és el de l'etiqueta
        info["control_clar"] = is_dead | ((inst_seq != NOBODY) & (inst_seq == state_seq))

    labels = np.full(n_frames, "D", dtype="<U1")
    labels[state_seq == HOME] = "H"
    labels[state_seq == AWAY] = "A"
    if cfg["fix_halves"]:
        labels[~graded] = "X"
    else:
        labels[:half1_start] = "X"
        labels[half1_end:half2_start] = "X"
        labels[half2_end:] = "X"

    if cfg.get("retime"):
        from starter.contact_v1 import exclusive_team
        from starter.timing_v1 import retime
        alive = graded & (labels != "D") & (labels != "X")
        per_frame = exclusive_team(kinematics["home_min_kappa"], kinematics["away_min_kappa"],
                                   alive, cfg["kappa"])
        labels, moved = retime(labels, per_frame, cfg["retime"])
        info["retimed_turnovers"] = len(moved)
    return labels, info


def run_pipeline(game="BAR-ATH", out_csv="evaluation/results/principled_v1.csv", data_dir=None,
                 config="p3"):
    cfg = CONFIGS[config]
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = games.source_frames("tactical", game)
    pitch = games.GAMES[game]["pitch"]
    half_blocks = games.half_video_blocks(game)
    k2 = games.GAMES[game]["kickoff_2nd"]
    kickoff_2_frame = (k2["minute"] * 60 + k2["second"]) * FPS + k2["frame"]

    # Same extraction code as markius_v1, so the same cache.
    cache_path = os.path.join(ROOT, "cache", "kinematics_v10.npz")
    kinematics = extract_or_load_kinematics(raw_path, meta_path, n_frames, pitch, cache_path=cache_path)

    st = {"home": None, "away": None, "set_pieces": None}
    if cfg["visible_dead"] == "st_veto" or cfg["init"] == "v1":
        st_path = games.asset_paths(game, data_dir)["smart_tagging"]
        st["home"], st["away"], st["set_pieces"] = load_smart_tagging_phases(st_path, n_frames)

    labels, info = predict(kinematics, st, n_frames, half_blocks, kickoff_2_frame, pitch, cfg)
    for k, v in info.items():
        print(f"{config}: {k} read off the match: {v}")

    out_path = out_csv if os.path.isabs(out_csv) else os.path.join(ROOT, out_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("frame,label\n")
        f.writelines(f"{i},{lbl}\n" for i, lbl in enumerate(labels))
    return out_path, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Principled v1 possession model")
    ap.add_argument("--config", default="p3", choices=sorted(CONFIGS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()
    out = args.out or ("evaluation/results/principled_v1.csv" if args.config == "p3"
                       else f"evaluation/results/principled_{args.config.replace('+', '_')}.csv")
    print(run_pipeline(out_csv=out, data_dir=args.data_dir, config=args.config)[0])
