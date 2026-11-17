"""Velocidad del balón en el balón muerto visible (figura 3, panel derecho; secciones 3 y 7 del documento técnico).
Frames con el balón visible donde las reglas y la TCN dicen balón muerto: mediana de la velocidad del balón en
  - el partido proporcionado (sin etiquetas): señales guardadas por `predict.py --guarda`;
  - el banco v8.2 (semillas 0-2 del banco v10) y el banco del equipo (semillas 3-5), IDSSE.
    MTAH_DEG=v10 ULISES_PROBS_SUF=_sim python diag_vel_d.py <senyals.npz del partido>"""
import sys
import numpy as np
import lab
from lab import *
q = lambda a: np.percentile(a, [10, 25, 50, 75, 90]).round(2)
z = np.load(sys.argv[1]); nn = [str(n) for n in z["noms"]]; v = z["X"][:, nn.index("pilota_vel")]
s = (z["period"] > 0) & z["r"] & (z["janv1"].astype("<U1") == "D") & (z["p_tcn"].argmax(1) == 2)
print(f"partido: n {int(s.sum())}  velocidad q10..q90 {q(v[s])}  mediana {np.median(v[s]):.2f}  frac < 0,3 m/s {np.mean(v[s] < 0.3):.3f}")
for nom, lls in (("banco v8.2 (s0-2)", (0, 1, 2)), ("banco del equipo (s3-5)", (3, 4, 5))):
    V = []
    for font, p in SELECCIO[:7]:
        for ll in lls:
            e = entrada("idsse", p, ll); X = feats(e); vv = X[:, e["noms"].index("pilota_vel")]
            ss = (e["period"] > 0) & e["r"].astype(bool) & (e["janv1"] == "D") & (e["p"].argmax(1) == 2); V.append(vv[ss]); e.pop("X", None)
    V = np.concatenate(V); print(f"{nom}: n {len(V)}  q10..q90 {q(V)}  mediana {np.median(V):.2f}  frac < 0,3 m/s {np.mean(V < 0.3):.3f}")
