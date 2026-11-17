#!/usr/bin/env bash
# E21: stack t2 entrenado con las 6 semillas del banco v12 (C11 en las dos mitades) y la TCN v10 simétrica.
cd "$(dirname "$0")"; V=../../.venv/bin/python; export ULISES_PROBS_SUF=_sim
until ! pgrep -f "stack.py t2 0" >/dev/null; do sleep 20; done
export MTAH_DEG=v12t10
for ll in "0,1,2" "3,4,5"; do export ULISES_LLAVORS=$ll ULISES_SUF="_sim_s${ll//,/}"; echo "== banco v12 (C11), semillas $ll"
  $V lab.py | grep -A1 "== base_fusio"; $V exp_stack.py t2 g3 | grep -E "^==|IDSSE acc|Δacc vs"; done
echo "== banco v10 sin C11, frente a model_v10_sim"
export MTAH_DEG=v10
for ll in "0,1,2" "3,4,5"; do export ULISES_LLAVORS=$ll ULISES_SUF="_sim_s${ll//,/}"
  $V xeval.py cache_v12t10_sim/stack_t2 e21t2 tcn,jv,raw,touch > /dev/null; $V exp_stack.py e21t2 g3 | grep -E "^==|IDSSE acc"
  $V compara_res.py results_v10_sim_s${ll//,/}/stack_t2_g3.json results_v10_sim_s${ll//,/}/stack_e21t2_g3.json; done
echo "== partido (sin etiquetas)"
export MTAH_DEG=v12t10 ULISES_LLAVORS=0,1,2 ULISES_SUF=_sim_s012
$V diag_d.py t2 tcn,jv,raw,touch 2>&1 | grep -E "variante|^D (TCN|stack|final) \| visible|stack D \| TCN|recall D final|precisión D final"
echo FIN
