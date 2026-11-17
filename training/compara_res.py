"""Compara dos resultados del laboratorio, pareado por partido (b − a), con las tres etiquetas y por fuente.
    MTAH_DEG=v3 python compara_res.py results_v8/stack_t2_g3.json results_v8_sim/stack_t2_g3.json [...más b]"""
import sys, json
import numpy as np
import lab
from lab import bootstrap
a = json.load(open(sys.argv[1]))
for fb in sys.argv[2:]:
    b = json.load(open(fb)); ks = sorted(a["partits"])
    out = []
    for k in ("acc", "dur", "prov"):
        d = [b["partits"][x][k] - a["partits"][x][k] for x in ks]
        m, lo, hi = bootstrap(d)
        di = [v for x, v in zip(ks, d) if x.startswith("idsse")]; dm = [v for x, v in zip(ks, d) if x.startswith("metrica")]
        out.append(f"{k} {b[k][0]:.4f} ({m:+.4f} [{lo:+.4f},{hi:+.4f}] {sum(v > 0 for v in d)}/10; IDSSE {np.mean(di):+.4f}, Metrica {np.mean(dm):+.4f})")
    print(f"{fb}\n   " + "\n   ".join(out))
