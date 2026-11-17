"""E18: ¿dónde se pierde el balón muerto? Compara banco (con etiquetas externas) y partido (SIN etiquetas) para la v10 simétrica.
Por frame, en las partes: fracción de D según reglas, TCN (argmax), stack (argmax) y salida decodificada, separando balón
visible / oculto, y probabilidad de que cada etapa conserve la D de la anterior.
    MTAH_DEG=v10 ULISES_PROBS_SUF=_sim python diag_d.py"""
import json, os, numpy as np, lab
from lab import *
def taula(nom, jv, pt, ps, y, r, m, g=None):
    o = {"n": int(m.sum())}
    D = lambda a: (a == 2)
    for et, a in (("reglas", jv), ("TCN", pt), ("stack", ps), ("final", y)):
        o[f"D {et}"] = float(D(a)[m].mean()); o[f"D {et} | visible"] = float(D(a)[m & r].mean()); o[f"D {et} | oculto"] = float(D(a)[m & ~r].mean())
    o["stack D | TCN D"] = float(D(ps)[m & D(pt)].mean()); o["final D | stack D"] = float(D(y)[m & D(ps)].mean())
    o["final D | reglas D"] = float(D(y)[m & D(jv)].mean()); o["final D | reglas D y TCN D"] = float(D(y)[m & D(jv) & D(pt)].mean())
    o["p_stack(D) media | reglas D"] = None
    if g is not None:
        ok = m & (g < 3)
        o["D real"] = float(D(g)[ok].mean()); o["D real | visible"] = float(D(g)[ok & r].mean()); o["D real | oculto"] = float(D(g)[ok & ~r].mean())
        for et, a in (("reglas", jv), ("TCN", pt), ("stack", ps), ("final", y)):
            tp = float((D(a) & D(g) & ok).sum()); o[f"recall D {et}"] = tp / max(float((D(g) & ok).sum()), 1); o[f"precisión D {et}"] = tp / max(float((D(a) & ok).sum()), 1)
    return o


def decod(e, p, cfg, st):
    prior = np.array(st["prior"]); em = cfg["a"] * (np.log(np.maximum(p, 1e-6)) - np.log(prior)[None]); em = em.copy()
    em[:, 2] += cfg.get("d", 0) * (e["janv1"] == "D")
    m, lq, ls = hmm(st, cfg["s"], cfg.get("sd")); return per_parts(e, lambda a, b: viterbi(em[a:b], m, lq, ls, np.log(prior)))


if __name__ == "__main__":
    # uso: diag_d.py [variante=t2] [grupos=tcn,jv,raw,touch]
    import sys, lightgbm as lgb, feats_stack as fs
    var = sys.argv[1] if len(sys.argv) > 1 else "t2"; g = tuple((sys.argv[2] if len(sys.argv) > 2 else "tcn,jv,raw,touch").split(","))
    par = json.load(open("model_v10_sim/params.json")); cfg, st = par["cfg"], par["st"]
    rf = os.path.join(lab.RESULTS, f"stack_{var}_g3.json")
    if os.path.exists(rf):      # configuración elegida para el partido (pliegue tots) en este banco
        cfg = json.load(open(rf))["tria"]["tots"]["cfg"]; st = stats("tots")
    grups = {}
    for font, p in SELECCIO:
        for ll in lab.LLAVORS:
            e = entrada(font, p, ll); ps = np.load(f"{lab.CACHE}/stack_{var}/{font}/{p}_s{ll}.npy").astype(float)
            y = decod(e, ps, cfg, st); m = e["period"] > 0
            grups.setdefault(f"{font} s{'012' if ll < 3 else '345'}", []).append(taula("", pm.label_codes(e["janv1"]), e["p"].argmax(1), ps.argmax(1), y, e["r"].astype(bool), m, e["gt_acc"]))
    z = np.load("cache_v10_sim/partit/senyals.npz")
    e = {"X": z["X"], "noms": [str(n) for n in z["noms"]], "p": np.round(z["p_tcn"].astype(np.float64), 5), "janv1": z["janv1"].astype("<U1"), "clar": z["clar"]}
    e["phys_conf"] = np.load("cache_v10_sim/partit/phys.npy") if os.path.exists("cache_v10_sim/partit/phys.npy") else None
    mdl = lgb.Booster(model_file=f"{lab.CACHE}/stack_{var}/model_tots.txt")
    Fa, names = fs.build(e, False, g); Fb, _ = fs.build(e, True, g); assert names == mdl.feature_name()
    psm = (mdl.predict(Fa) + mdl.predict(Fb)[:, [1, 0, 2]]) / 2
    e["period"] = z["period"]; ym = decod(e, psm, cfg, st)
    np.save(f"cache_v10_sim/partit/y_{var}.npy", ym)
    part = taula("", pm.label_codes(e["janv1"]), z["p_tcn"].argmax(1), psm.argmax(1), ym, z["r"], z["period"] > 0)
    cols = {k: {c: float(np.mean([x[c] for x in v])) for c in v[0] if v[0][c] is not None} for k, v in grups.items()}; cols["BAR-ATH (sin etiquetas)"] = part
    noms = list(cols); print(f"variante {var} {g}"); print(f"{'':30s}" + "".join(f"{n:>26s}" for n in noms))
    for c in cols[noms[0]]:
        if c == "n": continue
        print(f"{c:30s}" + "".join(f"{cols[n].get(c, float('nan')):26.3f}" for n in noms))
    json.dump(cols, open(f"results/diag_d_E18_{var}.json", "w"), indent=1)
