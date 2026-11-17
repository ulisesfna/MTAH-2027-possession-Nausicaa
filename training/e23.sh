#!/usr/bin/env bash
# E23: evaluación del stack con reglas sin retiming (p2_notag) frente a E21 (lo entregado), con el criterio del REPORT.
cd "$(dirname "$0")"; V=../../.venv/bin/python; H=~/mtah_data; export ULISES_PROBS_SUF=_sim
until ! pgrep -f "stack.py t2 0" >/dev/null && ! pgrep -f "[e]23_reglas_p2.py" >/dev/null; do sleep 20; done
echo "######## banco v12 (con C11): E23 frente a E21"
export MTAH_DEG=v12t10p2
for ll in "0,1,2" "3,4,5"; do s=${ll//,/}; export ULISES_LLAVORS=$ll ULISES_SUF="_sim_s$s"
  $V exp_stack.py t2 g3 | grep -E "^==|IDSSE acc"; $V compara_res.py results_v12t10_sim_s$s/stack_t2_g3.json results_v12t10p2_sim_s$s/stack_t2_g3.json; done
echo "######## (b) banco v10 semillas 0-2 (sin C11): E23 frente a E21"
export MTAH_DEG=v10p2
for ll in "0,1,2" "3,4,5"; do s=${ll//,/}; export ULISES_LLAVORS=$ll ULISES_SUF="_sim_s$s"
  $V xeval.py cache_v12t10p2_sim/stack_t2 e23 tcn,jv,raw,touch > /dev/null
  [ $s = 012 ] && { $V exp_stack.py e23 g3 | grep -E "^==|IDSSE acc"; $V compara_res.py results_v10_sim_s012/stack_e21t2_g3.json results_v10p2_sim_s012/stack_e23_g3.json; }; done
echo "######## (a) banco de referencia del equipo (vista v3view): E23 frente a E21 (0.7617)"
VW=$H/v3viewp2; C=cache_v3viewp2_sim/stack_c11; rm -rf $VW cache_v3viewp2_sim cache_v3viewp2 $H/bench_v3viewp2
mkdir -p $VW/features/{idsse,metrica} $VW/decoder/entrades/{idsse,metrica} $VW/tcn_probs_sim/{idsse,metrica} $C/{idsse,metrica} cache_v3viewp2/gt_prov/{idsse,metrica} $H/bench_v3viewp2/{idsse,metrica}
cp $H/v10/decoder/estadistiques_*.json $VW/decoder/
for k in 0 1 2; do j=$((k + 3))
  for f in $H/v10/features/idsse/*_s$j.npz; do b=$(basename $f _s$j.npz)
    ln -sfn $(readlink -f $f) $VW/features/idsse/${b}_s$k.npz
    ln -sfn $(readlink -f $H/v10p2/decoder/entrades/idsse/${b}_s$j.npz) $VW/decoder/entrades/idsse/${b}_s$k.npz
    ln -sfn $(readlink -f $H/v10/tcn_probs_sim/idsse/${b}_s$j.npz) $VW/tcn_probs_sim/idsse/${b}_s$k.npz
    ln -sfn $(readlink -f cache_v10p2_sim/stack_e23/idsse/${b}_s$j.npy) $C/idsse/${b}_s$k.npy
    ln -sfn $(readlink -f cache_v10/gt_prov/idsse/${b}_s$j.npy) cache_v3viewp2/gt_prov/idsse/${b}_s$k.npy
    ln -sfn $(readlink -f $H/bench_v10/idsse/${b}_s$j.npz) $H/bench_v3viewp2/idsse/${b}_s$k.npz
  done
  for f in $H/v10/features/metrica/*_s$k.npz; do b=$(basename $f .npz)
    ln -sfn $(readlink -f $f) $VW/features/metrica/$b.npz
    ln -sfn $(readlink -f $H/v10p2/decoder/entrades/metrica/$b.npz) $VW/decoder/entrades/metrica/$b.npz
    ln -sfn $(readlink -f $H/v10/tcn_probs_sim/metrica/$b.npz) $VW/tcn_probs_sim/metrica/$b.npz
    ln -sfn $(readlink -f cache_v10p2_sim/stack_e23/metrica/$b.npy) $C/metrica/$b.npy
    ln -sfn $(readlink -f cache_v10/gt_prov/metrica/$b.npy) cache_v3viewp2/gt_prov/metrica/$b.npy
    ln -sfn $(readlink -f $H/bench_v10/metrica/$b.npz) $H/bench_v3viewp2/metrica/$b.npz
  done
done
export MTAH_DEG=v3viewp2 ULISES_LLAVORS=0,1,2; unset ULISES_SUF
$V exp_stack.py c11 g3 | grep -E "^==|IDSSE acc"
$V compara_res.py results_v3view_sim/stack_c11_g3.json results_v3viewp2_sim/stack_c11_g3.json
echo "######## (c) partido sin etiquetas"
ln -sfn $H/v10/tcn_v10_weights.pt $H/v12t10p2/tcn_v12t10p2_weights.pt
MTAH_DEG=v12t10p2 ULISES_LLAVORS=0,1,2 ULISES_SUF=_sim_s012 ULISES_REGLES="janv1 p2_notag" $V export_model.py t2 stack_t2_g3 tcn,jv,raw,touch | tail -1
rm -rf model_v10c11p2_sim; mv model_v12t10p2_sim model_v10c11p2_sim
.venv_inf/bin/python predict.py --model model_v10c11p2_sim --out out/ulises_v10c11p2_sim.csv --guarda cache_v12t10p2_sim/partit_senyals.npz | grep sha256
$V - <<'PY'
import numpy as np
for nom, f in (("E21 (entregado)", "cache_v12t10_sim/partit_senyals.npz"), ("E23 (sin retiming)", "cache_v12t10p2_sim/partit_senyals.npz")):
    z = np.load(f); m = z["period"] > 0; r = z["r"] & m; jv = z["janv1"].astype("<U1"); pt = z["p_tcn"].argmax(1); ps = z["p_stack"].argmax(1); y = z["y"]
    print(f"{nom:20s} D total reglas/TCN/final {np.mean(jv[m]=='D'):.3f}/{np.mean(pt[m]==2):.3f}/{np.mean(y[m]==2):.3f} | D visible reglas/TCN/stack/final {np.mean((jv=='D')&r)/r.mean():.3f}/{np.mean((pt==2)&r)/r.mean():.3f}/{np.mean((ps==2)&r)/r.mean():.3f}/{np.mean((y==2)&r)/r.mean():.3f} | stack conserva D de la TCN {np.mean(ps[m&(pt==2)]==2):.3f}")
PY
echo FIN
