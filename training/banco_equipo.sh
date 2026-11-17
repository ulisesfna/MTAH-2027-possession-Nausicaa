#!/usr/bin/env bash
# Reproduce la cifra principal del documento técnico (spell-F1 0.762 en el banco de referencia del equipo) y la
# comparación con v9m1 (sección 2, figura 2). Requiere haber ejecutado antes e21_prepara.sh y e21.sh (deja las
# predicciones del stack entregado sobre el banco v10 en cache_v10_sim/stack_e21t2).
# El banco de referencia del equipo = IDSSE con su degradación (semillas 3-5 del banco v10) + Metrica con su degradación
# (semillas 0-2 del banco v10). Aquí se monta como "v3view" con enlaces y se evalúa con el protocolo anidado.
#   bash banco_equipo.sh <informe JSON del arnés del equipo (rama v9m1-regles)>
set -euo pipefail
cd "$(dirname "$0")"; V=${PY_LAB:-python}; H=${MTAH_LOCAL:-~/mtah_data}; H=$(eval echo $H)
VW=$H/v3view; C=cache_v3view_sim/stack_c11
rm -rf $VW cache_v3view_sim cache_v3view
mkdir -p $VW/features/{idsse,metrica} $VW/decoder/entrades/{idsse,metrica} $VW/tcn_probs_sim/{idsse,metrica} $C/{idsse,metrica} cache_v3view/gt_prov/{idsse,metrica} $H/bench_v3view/{idsse,metrica}
cp $H/v10/decoder/estadistiques_*.json $VW/decoder/
for k in 0 1 2; do j=$((k + 3))
  for f in $H/v10/features/idsse/*_s$j.npz; do b=$(basename $f _s$j.npz)
    ln -sfn $(readlink -f $f) $VW/features/idsse/${b}_s$k.npz
    ln -sfn $(readlink -f $H/v10/decoder/entrades/idsse/${b}_s$j.npz) $VW/decoder/entrades/idsse/${b}_s$k.npz
    ln -sfn $(readlink -f $H/v10/tcn_probs_sim/idsse/${b}_s$j.npz) $VW/tcn_probs_sim/idsse/${b}_s$k.npz
    ln -sfn $(readlink -f cache_v10_sim/stack_e21t2/idsse/${b}_s$j.npy) $C/idsse/${b}_s$k.npy
    ln -sfn $(readlink -f cache_v10/gt_prov/idsse/${b}_s$j.npy) cache_v3view/gt_prov/idsse/${b}_s$k.npy
    ln -sfn $(readlink -f $H/bench_v10/idsse/${b}_s$j.npz) $H/bench_v3view/idsse/${b}_s$k.npz
  done
  for f in $H/v10/features/metrica/*_s$k.npz; do b=$(basename $f .npz)
    ln -sfn $(readlink -f $f) $VW/features/metrica/$b.npz
    ln -sfn $(readlink -f $H/v10/decoder/entrades/metrica/$b.npz) $VW/decoder/entrades/metrica/$b.npz
    ln -sfn $(readlink -f $H/v10/tcn_probs_sim/metrica/$b.npz) $VW/tcn_probs_sim/metrica/$b.npz
    ln -sfn $(readlink -f cache_v10_sim/stack_e21t2/metrica/$b.npy) $C/metrica/$b.npy
    ln -sfn $(readlink -f cache_v10/gt_prov/metrica/$b.npy) cache_v3view/gt_prov/metrica/$b.npy
    ln -sfn $(readlink -f $H/bench_v10/metrica/$b.npz) $H/bench_v3view/metrica/$b.npz
  done
done
export MTAH_DEG=v3view ULISES_PROBS_SUF=_sim ULISES_LLAVORS=0,1,2
$V lab.py | grep -A1 "== base_fusio"
$V exp_stack.py c11 g3 | grep -E "^==|IDSSE acc|Δ"           # -> results_v3view_sim/stack_c11_g3.json: 0.7617
MTAH_DEG=v3 $V compara_v9m1.py results_v3view_sim/stack_c11_g3.json "$1" ../fig/fig_data.json
