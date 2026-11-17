#!/usr/bin/env python3
"""Contact v1: possession changes decided by sequences of contacts.

On the public 5 minutes, p2 places stoppage boundaries within about a second but turnovers 2-3 s
late, misses the shortest possessions and invents a change when a defender stays close to the ball
carrier. All three follow from one rule: a turnover needs 11 consecutive frames of the opponent
alone within reach. An interception followed by a quick pass never gives that.

A contact is a run of frames where one team alone is within reach of the ball (the same reach and
tie-break as markius_v1, from the kinematics cache). Then, inside every stretch of live ball:

  - a contact of at least K frames is control, and changes possession on its own;
  - a shorter contact by the other team only starts a candidate change;
  - the candidate is confirmed by that team's next contact (receive and pass, or pass received);
  - it is cancelled if the team in possession has a contact first (a deflection, a duel lost);
  - a confirmed change is dated to the first contact of the new team.

The touch logic is touch_v1.assign; only the events differ. Dead ball comes from p2 unchanged.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from starter.touch_v1 import AWAY, HOME, assign  # noqa: E402

DEFAULT = {
    "reach": "kappa",   # "kappa": markius_v1's reach model; or a fixed radius in metres
    "kappa": 0.95,      # markius_v1's declared threshold
    "min_contact": 3,   # frames; shorter runs are tracking flicker (0.12 s)
    "control": 11,      # frames; a contact this long is control (the Otsu value of principled_v1)
}


def exclusive_team(h_eff, a_eff, alive, kappa):
    """Per frame HOME, AWAY or -1: the team alone within reach, or the nearer one when both are."""
    with np.errstate(invalid="ignore"):
        hc = ~np.isnan(h_eff) & (h_eff <= kappa) & alive
        ac = ~np.isnan(a_eff) & (a_eff <= kappa) & alive
    out = np.full(len(h_eff), -1, dtype=np.int8)
    out[hc & ~ac] = HOME
    out[ac & ~hc] = AWAY
    both = hc & ac
    out[both & (h_eff < a_eff - 1e-4)] = HOME
    out[both & (a_eff < h_eff - 1e-4)] = AWAY
    return out


def exclusive_by_radius(d_home, d_away, alive, radius):
    with np.errstate(invalid="ignore"):
        h = (d_home <= radius) & ~(d_away < d_home) & alive
        a = (d_away <= radius) & ~(d_home < d_away) & alive
    out = np.full(len(d_home), -1, dtype=np.int8)
    out[h] = HOME
    out[a & ~h] = AWAY
    return out


def contacts(team_per_frame, min_contact, control):
    """[(start, team, kind)] for every run of one team; kind "C" if it lasts >= control frames."""
    events = []
    for team in (HOME, AWAY):
        mask = team_per_frame == team
        edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if e - s >= min_contact:
                events.append((int(s), team, "C" if e - s >= control else "T"))
    return events


def predict(base_labels, graded, cfg=None, h_eff=None, a_eff=None, d_home=None, d_away=None):
    """Relabel the live stretches of base_labels (p2) from contacts."""
    cfg = {**DEFAULT, **(cfg or {})}
    alive = graded & (base_labels != "D") & (base_labels != "X")
    if cfg["reach"] == "kappa":
        per_frame = exclusive_team(h_eff, a_eff, alive, cfg["kappa"])
    else:
        per_frame = exclusive_by_radius(d_home, d_away, alive, float(cfg["reach"]))
    events = contacts(per_frame, cfg["min_contact"], cfg["control"])
    touches = [(t, k) for t, k, kind in events if kind == "T"]
    controls = [(t, k) for t, k, kind in events if kind == "C"]
    labels, changes = assign(base_labels, touches, controls, graded)
    info = {"short_contacts": len(touches), "controls": len(controls), "changes": changes}
    return labels, info, per_frame
