#!/usr/bin/env bash
# E20: evalúa los stacks entrenados en el banco v11 (IDSSE v8 + balón muerto visible en movimiento, C11).
cd "$(dirname "$0")"; V=../../.venv/bin/python
until ! pgrep -f "stack.py t2(nv)? 0" >/dev/null; do sleep 20; done
echo "######## banco v11 (semillas 0-2)"
export MTAH_DEG=v11 ULISES_PROBS_SUF=_sim ULISES_LLAVORS=0,1,2
$V lab.py | grep -A1 "== base_fusio"
for nom in t2 t2nv; do $V exp_stack.py $nom g3 | grep -E "^==|IDSSE acc|Δacc vs"; done
$V compara_res.py results_v11_sim/stack_t2_g3.json results_v11_sim/stack_t2nv_g3.json
echo "######## stacks v11 evaluados en el banco v10 semillas 0-2 (sin C11), frente al stack de model_v10_sim"
export MTAH_DEG=v10 ULISES_SUF=_sim_s012
$V xeval.py cache_v11_sim/stack_t2 v11t2 tcn,jv,raw,touch > /dev/null
$V xeval.py cache_v11_sim/stack_t2nv v11t2nv tcn,jv,raw,touch,-vel > /dev/null
for nom in v11t2 v11t2nv; do $V exp_stack.py $nom g3 | grep -E "^==|IDSSE acc"; done
$V compara_res.py results_v10_sim_s012/stack_t2_g3.json results_v10_sim_s012/stack_v11t2_g3.json results_v10_sim_s012/stack_v11t2nv_g3.json
echo "######## comportamiento en el partido (sin etiquetas)"
export MTAH_DEG=v11; unset ULISES_SUF
$V diag_d.py t2 tcn,jv,raw,touch 2>&1 | grep -E "variante|^D (TCN|stack|final) \| visible|stack D \| TCN|final D \| reglas D y|recall D final|precisión D final"
$V diag_d.py t2nv tcn,jv,raw,touch,-vel 2>&1 | grep -E "variante|^D (TCN|stack|final) \| visible|stack D \| TCN|final D \| reglas D y|recall D final|precisión D final"
echo FIN
