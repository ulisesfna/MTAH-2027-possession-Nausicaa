#!/usr/bin/env bash
# Reentreno completo de TCN + decodificador + ulises sobre un banco propio (dataset_v8). Todo en ~/mtah_data/<deg>.
#   nohup bash run_banc.sh v8 > run_v8.log 2>&1 &
# Pasos (tiempos medidos en esta VM): features ~20 min · TCN 8 pliegues ~2,8 h · entradas ~10 min · base ~5 min ·
# stack ~45 min · TCN final ~20 min · partido ~5 min. Nada escribe en /workspace/shared ni en Git.
set -euo pipefail
DEG=${1:-v8}; DES=${2:-1}   # DES: paso desde el que se retoma
cd "$(dirname "$0")"
P=/workspace/shared/venvs/bench/bin/python; V=../../.venv/bin/python
export MTAH_DEG=$DEG MTAH_TCN=$DEG MTAH_DATA_DIR=$(cd ../.. && pwd)/data
t0=$(date +%s); pas() { echo; echo "== $(date +%H:%M:%S) ($(( ($(date +%s) - t0) / 60 )) min) $*"; }
B=base/bench
if [ $DES -le 1 ]; then pas "1 features"; (cd base && $P bench/features.py --src extern --jobs 2 | tail -2); fi
if [ $DES -le 2 ]; then pas "2 TCN cv (8 pliegues)"; (cd base && $P bench/train_tcn.py cv | grep -v "^   "); fi
if [ $DES -le 3 ]; then pas "3 entradas decoder"; (cd base && $P bench/tune_decoder.py entrades --jobs 2 | tail -1 && $P -c "
import sys; sys.path.insert(0,'bench'); sys.path.insert(0,'.')
import tune_decoder as td
for n in td.plecs(): td.estadistiques_plec(n)
print('estadísticas ok')"); fi
if [ $DES -le 4 ]; then pas "4 base fusio"; $V lab.py | tail -12; fi
if [ $DES -le 5 ] && [ ! -e cache_$DEG/stack_t2/model_tots.txt ]; then pas "5 stack t2"; $V stack.py t2 0 tcn,jv,raw,touch | tail -3; fi
if [ $DES -le 6 ]; then pas "6 evaluación stack"; $V exp_stack.py t2 g3 | tail -12; fi
if [ $DES -le 7 ] && [ ! -e ~/mtah_data/$DEG/tcn_${DEG}_weights.pt ]; then pas "7 TCN final"; (cd base && $P bench/train_tcn.py final | tail -2); fi
if [ $DES -le 8 ]; then pas "8 partido"; $P match_entry.py; $V match_predict.py t2 stack_t2_g3 tcn,jv,raw,touch; $V make_csv.py t2 stack_t2_g3 tcn,jv,raw,touch; fi
pas "fin"
