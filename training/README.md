# training — where every number of the technical document comes from

Training ran on the team's workstation. External data is read from `MTAH_SHARED` (default `/workspace/shared`) and
`MTAH_LOCAL` (default `~/mtah_data`); neither is in this repository. `lab.py` and `tcn_inf_otro.py` use the shared code
in `../base` (or `ULISES_BASE`). To train the TCN, first apply `base_patches/` to `../base/bench/` (the `.diff` files
show each change; the patched files are next to them). Environment variables prefixed `ULISES_` are internal switches
of the scripts (seeds, suffixes, threads); experiment ids (E1–E23) refer to `../docs/REPORT.md` (Spanish).

| Number in the document | Script | Saved result (first value of `acc`) |
|---|---|---|
| 0.761 and the comparison with the best model without stacking (+0.059, 10/10; figure 2) | `e23.sh` (part a: same benchmark view as `banco_equipo.sh`) + `compara_v9m1.py` | `../results/v3viewp2_sim/stack_c11_g3.json` |
| Stage 7 model before the rule clean-up (0.762) | `banco_equipo.sh` → `exp_stack.py c11 g3` | `../results/v3view_sim/stack_c11_g3.json` |
| 0.696, 0.705, 0.753 (history figure) | `lab.py`, `stack.py`, `exp_stack.py` | `../results/bench_jan/base_fusio.json`, `stack_tcn0_g1.json`, `stack_t2_g3.json` |
| Stage 7 stack (training and acceptance criterion) | `e21_prepara.sh` (runs `stack.py` and `e21.sh`) | `../results/v12t10_sim_s012/`, `../results/v10_sim_s012/stack_e21t2_g3.json` |
| Final model (experiment E23: stage-7 stack retrained with the rule set `p2_notag`; −0.001 [−0.004, +0.002]; 92.9 %) | `e23_reglas_p2.py` (rule inputs), `stack.py` on bench `v12t10p2`, `e23.sh` (evaluation, export, match check) | `../results/v12t10p2_sim_s*/`, `../results/v10p2_sim_s012/`, `../results/v3viewp2_sim/` |
| 93 % vs 89 %, 12.7 % and 9.0 % (dead-ball figure, left) | `diag_d.py` (inside `e19.sh`, `e20.sh`, `e21.sh`); final model: last block of `e23.sh` | `../results/bench_jan/diag_d_E18*.json` |
| 1.41 / 0.38 / 0.00 m/s (dead-ball figure, right) | `diag_vel_d.py` | — |
| AUC 0.90 → 0.74 (IDSSE), 0.998 → 0.86 (SkillCorner); descriptor distance 0.42 → 0.15 | `../dataset/adv_ds.py`, `../dataset/calib8.py` | `../dataset/REPORT_DATASET.md` |
| Dropped ideas (section 7) | `exp_stack.py`, `e19.sh`, `e20.sh`, `e22_posesiones.py`, `adv.py` | `../docs/REPORT.md` |

Overall order: `../dataset` (degradation) → TCN training with `base_patches` → `run_banc.sh` / `e21_prepara.sh`
(stack and nested evaluation) → `export_model.py` (writes the model folder copied to `../models/nausicaa/`).
