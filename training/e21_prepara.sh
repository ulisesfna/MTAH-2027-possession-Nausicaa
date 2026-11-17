#!/usr/bin/env bash
# Preparación del stack ENTREGADO (E21): datos con el balón muerto visible en movimiento (C11) en las dos degradaciones,
# probabilidades de la TCN v10 simétrica sobre ellos, y entrenamiento del stack. Es la secuencia que se ejecutó el
# 17-nov (antes estaba repartida en comandos sueltos); los resultados intermedios son idénticos (comprobado: diferencia 0).
# Requisitos: banco v8.2 (dataset/degrade8.py + fix_equips_sc.py) en $MTAH_LOCAL/bench_v8, TCN v10 entrenada
# (base_patches + bench/train_tcn.py con MTAH_DEG=v10 MTAH_TCN=v10 TCN_LLAVORS=0,1,2,3,4,5 TCN_PLECS_SC=4 TCN_SIM=1) y
# el banco v10 montado con dataset/build_v10.py.
#   PY_BENCH=<python con torch> PY_LAB=<python con lightgbm> bash e21_prepara.sh
set -euo pipefail
cd "$(dirname "$0")"; P=${PY_BENCH:-python}; V=${PY_LAB:-python}
H=$(eval echo ${MTAH_LOCAL:-~/mtah_data}); SH=${MTAH_SHARED:-/workspace/shared}; BASE=${ULISES_BASE:-$(cd ../base && pwd)}
# 1) C11 sobre IDSSE: mitad v8.2 (-> bench_v11) y mitad del equipo (-> bench_v12j)
$P ../dataset/carry11.py aplica $H/bench_v8/idsse $H/bench_v11/idsse
$P ../dataset/carry11.py aplica $SH/bench_v7d_deg/idsse $H/bench_v12j/idsse
mkdir -p $H/bench_v11/metrica; for f in $H/bench_v8/metrica/*; do ln -sfn $(readlink -f $f) $H/bench_v11/metrica/$(basename $f); done
# 2) features y entradas del decodificador (código del equipo con base_patches)
for D in v11 v12j; do mkdir -p $H/$D/decoder; cp $H/v8/decoder/estadistiques_*.json $H/$D/decoder/
  (cd $BASE && MTAH_DEG=$D $P bench/features.py --src extern --jobs 2 --fonts idsse && MTAH_DEG=$D $P bench/tune_decoder.py entrades --jobs 2); done
mkdir -p $H/v11/features/metrica; for f in $H/v8/features/metrica/*; do ln -sfn $(readlink -f $f) $H/v11/features/metrica/$(basename $f); done
# 3) banco v12 (enlaces): semillas 0-2 = v11, 3-5 = v12j renumeradas
$P ../dataset/build_v12.py
# 4) TCN v10 simétrica (sin reentrenar) sobre los datos con C11 -> banco "v12t10"
T=$H/v12t10; mkdir -p $T/tcn_probs_sim/idsse $T/tcn_probs_sim/metrica
ULISES_BASE=$BASE MTAH_DEG=v11  $P tcn_inf_otro.py $H/v10/tcn_models $T/tcn_probs_sim 0 idsse
ULISES_BASE=$BASE MTAH_DEG=v12j $P tcn_inf_otro.py $H/v10/tcn_models $T/tcn_probs_sim 3 idsse
for f in $H/v10/tcn_probs_sim/metrica/*; do ln -sfn $(readlink -f $f) $T/tcn_probs_sim/metrica/$(basename $f); done
ln -sfn $H/v12/features $T/features; ln -sfn $H/v12/decoder $T/decoder; ln -sfn $H/bench_v12 $H/bench_v12t10
ln -sfn $H/v10/tcn_v10_weights.pt $T/tcn_v12t10_weights.pt
# 5) máscaras de label_prov (caché de etiquetas externas)
mkdir -p cache_v12t10; $P - <<'PY'
import os, sys, glob, numpy as np
B = os.environ.get("ULISES_BASE") or os.path.abspath("../base"); [sys.path.insert(0, p) for p in (B, B + "/bench", B + "/evaluation")]
import harness as hz
H = os.path.expanduser(os.environ.get("MTAH_LOCAL", "~/mtah_data"))
for font in ("idsse", "metrica"):
    for f in sorted(glob.glob(f"{H}/bench_v12/{font}/*.npz")):
        z = np.load(f, allow_pickle=False); g = hz.gt_avaluable({k: z[k] for k in ("label_prov", "valid", "period")}, "label_prov")[0]
        d = f"cache_v12t10/gt_prov/{font}"; os.makedirs(d, exist_ok=True); np.save(f"{d}/{os.path.basename(f)[:-4]}.npy", g.astype(np.int8))
PY
# 6) stack entregado: 7 IDSSE x 6 semillas, probabilidades simétricas
MTAH_DEG=v12t10 ULISES_PROBS_SUF=_sim ULISES_LLAVORS=0,1,2,3,4,5 ULISES_PAS=8 ULISES_FILS=2 $V stack.py t2 0 tcn,jv,raw,touch
# 7) evaluación (e21.sh) y empaquetado
bash e21.sh
MTAH_DEG=v12t10 ULISES_PROBS_SUF=_sim ULISES_LLAVORS=0,1,2 ULISES_SUF=_sim_s012 $V export_model.py t2 stack_t2_g3 tcn,jv,raw,touch
