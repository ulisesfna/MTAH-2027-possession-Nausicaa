# Nausicaa: ball possession from tracking data

**Team El Chato** · match BAR–ATH · model **Nausicaa** · input: ATD tracking only

*Nausicaa is named after the princess of Homer's* Odyssey *(Book VI) whose ball game on the beach is the oldest
ball game in Western literature; the ball is lost into the water, and that is how the story moves on.*

| <span class="big">0.761</span><br>Spell-F1 on 10 external matches (our best model without stacking: 0.701) | <span class="big">10 / 10</span><br>External matches improved over that model | <span class="big">ATD only</span><br>No video, no events, no labels of the match | <span class="big">Byte-exact</span><br>Same CSV on Linux ARM and Windows x86 |
|---|---|---|---|

## 1. Approach

<img src="FIGDIR/fig1_pipeline.png" width="660">

| Step | What it does | How | Why |
|---|---|---|---|
| **1 · Features** | Describes each frame with 35 numbers | Ball position, visibility, smoothed speed and acceleration, time since last seen, distance to the lines; for each team, distance and relative speed of its two closest players to the ball, centroid, spread, visible players; pitch oriented by attacking direction | A compact, team-symmetric view of who is near the ball |
| **2 · TCN** | Per-frame probabilities of H / A / D | Temporal convolutional network: 8 residual blocks of dilated convolutions (dilation 1 → 128), receptive field ±20 s; trained with cross-entropy and triple weight within ±0.5 s of a possession change; run twice at inference (as is and with the teams swapped) and averaged | Possession depends on what happened before and after the frame |
| **3 · Rules** | A deterministic H / A / D label | The closest player within reach becomes a candidate; possession changes when the candidate keeps the ball (low relative speed) for a few frames; dead ball when the ball leaves the pitch or play visibly stops; also a "clear control" flag | Transparent and strong on clear control; makes different mistakes than the TCN |
| **4 · Stack** | Corrected H / A / D probabilities | Gradient-boosted trees (LightGBM, 250 trees) reading the TCN output at nine instants from −2 s to +2 s and its 1, 3 and 6 s averages, the rule label around the frame and how long ago it changed, the rule confidence, and the local geometry (distances, ball speed, which team was last and will next be within 1–2 m of the ball) | Learns when to trust the TCN and when the rules or the geometry |
| **5 · Viterbi** | Possession spells | Hidden Markov model with three states and a minimum duration per state (from the training spell lengths); emissions are the stack probabilities; best path per half | The metric scores spells, not frames: avoids flickering |

**How the pieces fit together.** Two branches read the same tracking independently and never see each other: a
*learned branch* (features → TCN) and a *rule-based branch* (rules, which work directly on positions and speeds). The
stack is the only place where they meet: for every frame it receives both opinions in a window around the frame, plus
the local geometry, and outputs one set of probabilities. It is trained on external matches where the TCN
probabilities come from networks that did not see that match, so it learns how reliable each source really is. The
Viterbi decoder sees only the stack output. Frames outside the two halves are written as `X`.

## 2. Possession rules (what the model learns)

| Situation | Label |
|---|---|
| Ball out, foul, goal or offside, until the first touch of the restart | **Dead ball (D)** |
| Team B makes a controlled action (pass, carry, shot, keeper catch) | Possession changes to B **from its first touch** |
| Loose ball, challenge, deflection or block without control | Possession **stays** with the previous team |
| Restart (throw-in, free kick, corner, kick-off) | The restarting team, from the first touch |
| Spell shorter than the 5th percentile of the training spells | Not emitted (decoder) |
| Ball not tracked | **Not** a reason for D by itself |

Robustness was checked against two other conventions (1 s of close control; the provider's own possession channel).
Which one matches the official ground truth is unknown: this is our largest uncertainty.

## 3. Training data and evaluation benchmark

| Dataset | Matches | Role | Labelled by | Source · licence |
|---|---|---|---|---|
| **IDSSE** (Bassek et al., *Sci. Data* 12:195, 2025) | 7 · Bundesliga 2022/23 · optical tracking, ~11 h | Training + model selection | DFL / Sportec: ball status and possession per frame, official events | Figshare · CC-BY 4.0 (DFL) |
| **SkillCorner** Open Data | 20 · A-League 2024/25 · broadcast tracking | Training (TCN) | SkillCorner's automatic system | GitHub · MIT |
| **Metrica** Sports sample data | 3 · anonymised | **Test only** | Metrica events | GitHub · source credited |

Sources: IDSSE https://doi.org/10.6084/m9.figshare.28196177 · SkillCorner https://github.com/SkillCorner/opendata ·
Metrica https://github.com/metrica-sports/sample-data. No data from the provided match is used for training or
validation.

**How the data was worked** (in this order):

1. **Common format.** All matches converted to 25 fps, metres, centred pitch and the same orientation.
2. **Labels.** A versioned, deterministic script turns each provider's annotations into H / A / D per frame with the
   rules of section 2, plus the two alternative conventions.
3. **Degradation.** External tracking is much cleaner than the provided ATD, so it is degraded to look like it,
   calibrated on **aggregate distributions of the provided tracking only** (for example, how often and where the ball
   disappears, how noisy it is): no labels, and nothing tied to specific frames of the match. Each match is used with six
   random degradations and with the teams swapped.
4. **Validation without labels.** Three checks: distance between ~20 tracking descriptors; a classifier trying to tell
   external frames from provided ones (AUC, 0.5 = indistinguishable); and the behaviour of simple models on both.
5. **Benchmark.** The 7 IDSSE and 3 Metrica matches, scored with spell-F1. Each match is scored with parameters chosen
   on the other nine (nested selection), and models are compared with a paired bootstrap over matches. Every change was
   accepted only if it met a criterion written down before the experiment, under all three labelling conventions.

| Change to the external data | What it fixes | Effect |
|---|---|---|
| The ball disappears while moving near a player and reappears after a kick (it used to vanish standing still) | Ball gaps in the wrong places | Descriptor distance 0.42 → 0.15; AUC (IDSSE) 0.90 → 0.74 |
| Slow ball wander, player drop-outs during ball gaps, wrong-team detections | Too-clean ball and player tracks | Included in the figures above |
| SkillCorner: keep the provider's extrapolated players | 9.5 visible players instead of ~20.5 | AUC (SkillCorner) 0.998 → 0.86 |
| SkillCorner: team codes corrected | Teams were swapped relative to the labels | TCN frame accuracy on IDSSE 0.51–0.73 → 0.88–0.93 |
| Visible dead ball moves (carried to the restart spot) | The stack had learnt "moving ball = in play" | On the match the stack keeps 93 % of the TCN's dead ball instead of 89 % |

## 4. How we got here

Nausicaa was not designed in one go. It is the result of seven stages, each started by a measured weakness of the
previous one. Unless stated otherwise, every number is spell-F1 on the benchmark of section 3.

<img src="FIGDIR/fig4_historia.png" width="620">

The dashed line (0.701) is the stage-3 pipeline with refined rules, developed in parallel: our best model without
stacking, and the reference used throughout this document.

| Stage | Starting problem | What we built or tried | What we learned | Kept |
|---|---|---|---|---|
| **1 · Rules** | First, a transparent baseline | Hand-written physical rules (closest player, sequence of touches) decoded with hidden Markov and semi-Markov models with minimum durations (~6 versions) | External providers do not agree on what possession is, so every change must hold under several labelling conventions | The rules, later as one input among others |
| **2 · Per-frame learning** | Rules cannot weigh many weak cues at once | The common format for the 30 external matches; a per-frame classifier on hand-made features (~4 versions) | Per-frame models flicker and need temporal context; the quality of external labels matters as much as the model | The external dataset |
| **3 · Temporal network** | Possession depends on seconds of context | The degraded benchmark; a TCN with ±20 s of context, combined with the rules in a Viterbi decoder (~4 networks, ~7 decoders) | TCN + rules + Viterbi: **0.696** | The TCN, the decoder, the benchmark |
| **4 · Stacking** | Error analysis: spell-F1 is lost at possession changes with the ball visible, not in long ball gaps; tuning the decoder alone barely helped | A second-level model that sees the TCN, the rules and the local geometry around each frame (~12 versions) | **0.753**. With the TCN alone as input the stack gives only 0.705: the gain comes from combining sources, not from smoothing | The stack |
| **5 · Data that looks like the match** | A pipeline check on the public five minutes disagreed with the benchmark | Instead of tuning on that window (not allowed), we tested without labels whether the external data resembles the provided tracking. It did not; we rebuilt the degradation and found swapped team codes in one provider (section 3) | The model was limited by the data more than by its architecture | The new degradation and the team-code fix |
| **6 · Robustness** | Each model was best on the degradation it was trained on | TCN trained on two degradations at once; prediction averaged with the teams swapped | +0.013 on the corrected data (symmetric prediction alone: +0.010); no loss when the test degradation changes | Both |
| **7 · Dead ball** | On the provided match the stack discarded visible dead ball that both the TCN and the rules saw | Found the cause: in external matches the visible dead ball stands still, on the provided match it is carried to the restart spot (median 1.4 m/s). The degradation now moves it; the stack was retrained | No loss on the benchmark and more dead ball kept on the match (section 8) | **Final model** |

The thread: start from transparent rules; add learned temporal context; combine both instead of choosing one; and
when the remaining errors came from differences between external and provided tracking rather than from the model,
fix the data instead of adding complexity. Ideas that did not pass their acceptance test are listed in section 7.

## 5. Results on external matches

<img src="FIGDIR/fig2_resultats.png" width="600">

- Spell-F1 **0.761** [95 % CI 0.720–0.793] against 0.701 for our best model without stacking: **+0.059**
  [+0.042, +0.081], better on all 10 matches.
- On the 3 Metrica matches, never used for training: +0.034.
- On the provided match: 192 possessions, 19.1 s on average, 42 % dead ball.

## 6. How the provided match is used

Nothing was labelled or corrected by hand. No parameter, threshold or model of the submitted pipeline was fitted or
chosen on the labels of the provided match or on the score of its five public minutes; our decisions were measured on
the benchmark of section 3. The provided match enters the submission only in these ways:

- **Its tracking, without labels**, which the rules allow: the degradation is calibrated on its aggregate
  distributions (section 3), and the final stack had to pass one label-free check on it (keep more of the TCN's dead
  ball than the previous stack: 92.9 % against 89 %).
- **The five public minutes** were scored as a pipeline check. Once, they showed where to look: every variant predicted
  far less dead ball than the window contains. The problem was diagnosed and fixed on external matches and on the
  unlabelled tracking (stages 5 and 7), never on the window's score.

## 7. Assets: used, not used, dropped

| Used | Not used | Tried and dropped (effect on external matches) |
|---|---|---|
| ATD tracking (only input) | Video, homography | Decoder-only tuning (≤ +0.007) |
| IDSSE, SkillCorner, Metrica | Broadcast tracking | Direct rule weight in the decoder (−0.006 to −0.049) |
| PyTorch, LightGBM (CPU) | Smart Tagging events | "Kick" features (behave differently on the provided tracking) |
| | Any human label of the match | Stack also trained on SkillCorner (−0.016) |
| | | Removing ball speed from the stack (cost accuracy) |
| | | Retraining the TCN with the dead-ball fix (no gain) |
| | | Forcing the number of possessions in the decoder (no gain, even with the true count of each external match) |

## 8. Limitations, and what we did about them

<img src="FIGDIR/fig3_pilota_morta.png" width="600">

| Limitation | What we did | What remains |
|---|---|---|
| **Dead ball moves on the match but stands still in external data** | Stage 7 (figure above) | The stack still predicts less visible dead ball than the rules (12.7 % vs 14.8 %) |
| **Short possession changes are missed** | The stack recovers part of them; forcing more changes does not help | Fewer spells than reality |
| **External ≠ provided tracking** | Degradation calibrated on aggregates (section 3) | Still distinguishable (AUC 0.74–0.86) |
| **Few matches** (the stack is trained on 7) | Paired bootstrap, nested selection | Wide confidence intervals |
| **Second-level leakage** on IDSSE: when an IDSSE match is scored, the stack's training inputs on the other matches come from TCNs that saw it | Metrica, which has no leakage, is reported separately (section 5) | — |

With more time: a model of how a real tracker loses the ball; set-piece features; finding out which possession
convention the ground truth follows.

## 9. Reproduction

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock \
    --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python predict.py --data <data folder> --out submission.csv
```

- `<data folder>` is the folder of the competition data bundle that contains `tactical/BAR-ATH_tactical_atd.txt`
  and `.xml` (only the ATD tracking is read). About one minute; one CPU thread, fixed seeds.
- Expected sha256: `8c5523a89afa68d87764404e3ca85a1f381d99113f70b0cc7a5b905decce2d9d`.
- **Why sha256.** SHA-256 is a cryptographic hash function: it gives a 64-character fingerprint of a file, and changing a
  single label changes the whole fingerprint. Equal fingerprints therefore mean byte-identical files. We obtained the same
  fingerprint on Linux ARM, on Windows x86 and from a fresh clone of this repository. This shows reproducibility on
  the machines tested; it is backed by design: fixed seeds, deterministic algorithms, one thread, pinned library
  versions, and probabilities rounded to 5 decimals before the stack and the decoder, so that tiny floating-point
  differences between machines cannot change a label.
- `models/nausicaa/`: trained weights · `base/`: shared inference code (data loading, features, TCN, rules, decoder) ·
  `training/` and `dataset/`: training and evaluation code (published as code; training is not bit-reproducible
  across machines) · `results/`: benchmark results.
