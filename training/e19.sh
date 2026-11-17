#!/usr/bin/env bash
# E19: espera a los stacks t2nv (sin velocidad del balón) y t2tj (solo TCN + reglas), los evalúa en los dos grupos de
# semillas de v10 y saca el diagnóstico de balón muerto banco/partido (sin etiquetas del partido).
cd "$(dirname "$0")"; V=../../.venv/bin/python; export MTAH_DEG=v10 ULISES_PROBS_SUF=_sim
until ! pgrep -f "stack.py t2(nv|tj) 0" >/dev/null; do sleep 20; done
for ll in "0,1,2" "3,4,5"; do export ULISES_LLAVORS=$ll ULISES_SUF="_sim_s${ll//,/}"; echo "######## semillas $ll"
  for nom in t2nv t2tj; do $V exp_stack.py $nom g3 | grep -E "^==|IDSSE acc"; $V compara_res.py results_v10$ULISES_SUF/stack_t2_g3.json results_v10$ULISES_SUF/stack_${nom}_g3.json; done; done
export ULISES_LLAVORS=0,1,2; unset ULISES_SUF
$V diag_d.py t2nv tcn,jv,raw,touch,-vel 2>&1 | grep -E "variante|D TCN|D stack|D final|stack D|final D|D real|recall D (stack|final)|precisión D (stack|final)|D reglas"
$V diag_d.py t2tj tcn,jv 2>&1 | grep -E "variante|D stack|D final|stack D|final D"
echo FIN
