"""Agregados de visibilidad de jugadores del BAR-ATH para C8/C9 (sin etiquetas) -> target8_jug.json"""
import json, numpy as np
from common import *
import target8, sig

def estad(b, xy, per, team):
    m = per > 0; pres = np.isfinite(xy[..., 0]); n = pres.sum(1); vis = np.isfinite(b[:, 0])
    nt = [pres[:, team == k].sum(1) for k in (0, 1)]
    hh = lambda x: [float(np.mean(x == k)) for k in range(0, 23)] + [float(np.mean(x >= 23))]
    out = {"hist_n_vis": hh(n[m & vis]), "hist_n_ocult": hh(n[m & ~vis]), "n_mitja_vis": float(n[m & vis].mean()), "n_mitja_ocult": float(n[m & ~vis].mean())}
    for thr in (8, 14, 18, 20):
        r = sig.trams((n < thr) & m); d = np.array([e - a for a, e in r]) / 25
        out[f"lt{thr}"] = {"frac": float(np.mean(n[m] < thr)), "ratxes_h": len(r) / (m.sum() / 90000), "dur_q": sig.q(d, (25, 50, 75, 90)), "frac_ocult": float(np.mean(~vis[(n < thr) & m]))}
    e12 = ((nt[0] >= 12) | (nt[1] >= 12)) & m
    r = sig.trams(e12); d = np.array([e - a for a, e in r]) / 25
    out["equip12"] = {"frac": float(e12.sum() / m.sum()), "ratxes_h": len(r) / (m.sum() / 90000), "dur_q": sig.q(d, (25, 50, 75, 90)), "frac_n22": float(np.mean(n[e12] == 22)), "frac_n23": float(np.mean(n[e12] >= 23))}
    dn = np.abs(np.diff(n[m])); out["canvis_min"] = float(dn.sum() / (m.sum() / 1500))
    return out

if __name__ == "__main__":
    d = partit(); b, xy, per, camp = target8.centrat(d)
    t = estad(b, xy, per, np.asarray(d["team_of_track"]))
    json.dump(t, open(os.path.join(AQUI, "target8_jug.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in t.items() if not k.startswith("hist")}, indent=0))
    import glob
    for f in externs(V8, "idsse")[:1]:
        z = np.load(f); print("v8a:", json.dumps({k: v for k, v in estad(z["ball_xyz"][:, :2].astype(float), z["players_xy"].astype(float), z["period"], z["players_team"]).items() if not k.startswith("hist")}))
