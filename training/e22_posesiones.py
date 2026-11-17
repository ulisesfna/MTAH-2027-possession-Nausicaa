"""E22: ¿ayuda forzar el número de posesiones? Sobre el modelo entregado (model_v10c11_sim) en el banco de referencia del
equipo (vista v3view). Se añade un sesgo delta a todas las transiciones del Viterbi (más delta = más cambios) y se mide:
  - base: delta = 0 (lo entregado);
  - TECHO (trampa, solo para ver el margen): en cada partido, el delta cuyo número de posesiones H/A más se acerca al REAL;
  - LEGÍTIMO (anidado): el delta que iguala el ritmo de posesiones por minuto medio de los OTROS partidos IDSSE.
Nada usa el partido del concurso.   MTAH_DEG=v3view ULISES_PROBS_SUF=_sim python e22_posesiones.py"""
import json, numpy as np, lab
from lab import *
R = json.load(open(f"{lab.RESULTS}/stack_c11_g3.json"))["tria"]
DELTAS = np.round(np.arange(-3.0, 3.01, 0.25), 2)


def decod(e, p, cfg, st, delta):
    prior = np.array(st["prior"]); em = cfg["a"] * (np.log(np.maximum(p, 1e-6)) - np.log(prior)[None]); em = em.copy()
    em[:, 2] += cfg.get("d", 0) * (e["janv1"] == "D")
    m, lq, ls = hmm(st, cfg["s"]); ls = np.array(ls, float).copy()
    off = ~np.eye(3, dtype=bool); ls[off] = ls[off] + delta
    return per_parts(e, lambda a, b: viterbi(em[a:b], m, lq, ls, np.log(prior)))


def runs_of(e):
    return [(int(np.flatnonzero(e["period"] == h)[0]), int(np.flatnonzero(e["period"] == h)[-1]) + 1) for h in (1, 2) if (e["period"] == h).any()]


tab = {}
for font, p in SELECCIO:
    pl = plec_de(font, p); cfg = R[pl]["cfg"]; st = stats(pl)
    for ll in lab.LLAVORS:
        e = entrada(font, p, ll); ps = np.load(f"{lab.CACHE}/stack_c11/{font}/{p}_s{ll}.npy").astype(float)
        g = e["gt_acc"]; rr = runs_of(e); mins = (e["period"] > 0).sum() / 25 / 60
        n_real = len(lab._spells(np.where(g == 3, 2, g).astype(np.int8), rr))
        rows = []
        for dlt in DELTAS:
            y = decod(e, ps, cfg, st, dlt); rows.append((dlt, len(lab._spells(y, rr)), spell_f1(g, y)))
        tab[(font, p, ll)] = {"rows": rows, "n_real": n_real, "mins": mins}
    print(font, p, flush=True)

def mean_by_match(fn):
    out = {}
    for (font, p, ll), t in tab.items():
        out.setdefault((font, p), []).append(fn(font, p, t))
    return {k: float(np.mean(v)) for k, v in out.items()}

base = mean_by_match(lambda f, p, t: [r for r in t["rows"] if r[0] == 0][0][2])
techo = mean_by_match(lambda f, p, t: min(t["rows"], key=lambda r: abs(r[1] - t["n_real"]))[2])
rate = {(f, p): np.mean([t["n_real"] / t["mins"] for (ff, pp, ll), t in tab.items() if ff == "idsse" and pp != p]) for (f, p, ll) in tab}
legit = mean_by_match(lambda f, p, t: min(t["rows"], key=lambda r: abs(r[1] - rate[(f, p)] * t["mins"]))[2])
best = mean_by_match(lambda f, p, t: max(r[2] for r in t["rows"]))
ks = sorted(base)
n0 = mean_by_match(lambda f, p, t: [r for r in t["rows"] if r[0] == 0][0][1]); nr = mean_by_match(lambda f, p, t: t["n_real"])
print("posesiones predichas / reales (delta 0):", round(np.mean(list(n0.values())), 1), "/", round(np.mean(list(nr.values())), 1))
for nom, v in (("base (entregado)", base), ("TECHO nº real (trampa)", techo), ("legítimo: ritmo de los otros IDSSE", legit), ("mejor delta por partido (trampa)", best)):
    d = [v[k] - base[k] for k in ks]; m, lo, hi = bootstrap(d)
    print(f"{nom:38s} spellF1 {np.mean([v[k] for k in ks]):.4f}  Δ {m:+.4f} [{lo:+.4f},{hi:+.4f}] mejor en {sum(x > 1e-9 for x in d)}/10")
json.dump({n: {"/".join(k): v for k, v in d.items()} for n, d in (("base", base), ("techo", techo), ("legit", legit), ("best", best))}, open("results/e22_posesiones.json", "w"), indent=1)
