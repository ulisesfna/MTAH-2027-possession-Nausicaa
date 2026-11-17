import json, glob, sys, numpy as np
from common import *
import degrade8 as d8, target8_jug as tj
T = d8.TJG
fs = sorted(glob.glob(f"{NET}/idsse/*.npz"))[:int(os.environ.get("NF", 2))]
def mostra(nom, E):
    f = lambda k, kk: np.mean([e[k][kk] for e in E])
    print(f"{nom:34s} n|vis {np.mean([e['n_mitja_vis'] for e in E]):.2f} n|ocult {np.mean([e['n_mitja_ocult'] for e in E]):.2f} | lt8 {f('lt8','frac'):.3f} lt14 {f('lt14','frac'):.3f} ({f('lt14','ratxes_h'):.0f}/h) lt18 {f('lt18','frac'):.3f} lt20 {f('lt20','frac'):.3f} ({f('lt20','ratxes_h'):.0f}/h, ocult {f('lt20','frac_ocult'):.2f}) | equip12 {f('equip12','frac'):.3f} ({f('equip12','ratxes_h'):.0f}/h, n22 {f('equip12','frac_n22'):.2f}) | canvis/min {np.mean([e['canvis_min'] for e in E]):.1f}"
          f" | P(22|vis) {np.mean([e['hist_n_vis'][22] for e in E]):.2f} P(21|vis) {np.mean([e['hist_n_vis'][21] for e in E]):.2f} P(22|ocult) {np.mean([e['hist_n_ocult'][22] for e in E]):.2f} P(23+) {np.mean([e['hist_n_vis'][23] for e in E]):.3f}", flush=True)
mostra("BAR-ATH", [T])
for cfg in sys.argv[1:]:
    P = {**d8.PARAMS, **json.load(open("calib8_sondes.json"))["params"]}
    for kv in cfg.split(","):
        k, v = kv.split("="); P[k] = type(d8.PARAMS[k])(float(v))
    E = []
    for f in fs:
        d, camp = d8.degrada8(f, 0, P); E.append(tj.estad(d["ball_xyz"][:, :2].astype(float), d["players_xy"].astype(float), d["period"], d["players_team"]))
    mostra(cfg[:34], E)
