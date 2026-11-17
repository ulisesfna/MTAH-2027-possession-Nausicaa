#!/usr/bin/env python3
"""Re-date turnovers to the flight of the ball between the two teams.

On the public 5 minutes, where the ball travels from one team to the other, the ground truth
changes possession about halfway through the flight: between the last contact of the team losing
the ball and the first contact of the team winning it. p2 changes when the new team is in control,
which is where its systematic 2-3 s lag at turnovers comes from.

This takes any labelling and moves each direct H<->A change (not changes through a stoppage) to a
point of that flight. The rule has no tuned constant:

  - "leave":    the end of the losing team's last contact;
  - "midpoint": halfway between that and the start of the winning team's first contact;
  - "arrive":   the start of the winning team's first contact.

Contacts are frames where one team alone is within reach (contact_v1.exclusive_team). A flight
longer than MAX_FLIGHT frames is left alone: that is a lost ball rather than a ball in the air,
the same 5 s scale that separates the ball's short and long tracking gaps.
"""

import numpy as np

HOME, AWAY = 0, 1
MAX_FLIGHT = 125
LOOKBACK = 250      # frames searched back for the losing team's last contact (10 s)


def retime(labels, per_frame, rule="midpoint"):
    """Return a copy of labels with every direct H<->A change moved according to rule."""
    out = labels.copy()
    n = len(labels)
    team_of = {"H": HOME, "A": AWAY}
    moved = []
    change_frames = [i for i in range(1, n)
                     if labels[i] in team_of and labels[i - 1] in team_of and labels[i] != labels[i - 1]]
    for f in change_frames:
        old, new = team_of[labels[f - 1]], team_of[labels[f]]
        # The new team's contact that the change belongs to: the latest start of a run of the new
        # team at or before the end of the first new-team run after f - LOOKBACK.
        lo = max(0, f - LOOKBACK)
        window = per_frame[lo:min(n, f + MAX_FLIGHT)]
        old_idx = np.flatnonzero(window == old)
        old_idx = old_idx[old_idx + lo < f]
        if not len(old_idx):
            continue
        last_old = int(old_idx[-1]) + lo                     # last frame of the losing team's contact
        after = per_frame[last_old + 1:min(n, f + MAX_FLIGHT)]
        new_idx = np.flatnonzero(after == new)
        if not len(new_idx):
            continue
        first_new = last_old + 1 + int(new_idx[0])           # first frame of the winning team's contact
        if first_new - last_old > MAX_FLIGHT:
            continue
        target = {"leave": last_old + 1, "midpoint": (last_old + 1 + first_new) // 2,
                  "arrive": first_new}[rule]
        # Only move within the same live stretch: never across a D or X.
        a, b = sorted((f, target))
        if np.any((labels[a:b] == "D") | (labels[a:b] == "X")):
            continue
        letter_old, letter_new = labels[f - 1], labels[f]
        if target < f:
            out[target:f] = letter_new
        elif target > f:
            out[f:target] = letter_old
        moved.append(target - f)
    return out, moved
