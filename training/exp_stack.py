"""Evalúa un modelo apilado: sustituye p de la TCN por p del stack (cache/stack_<nombre>) en el decodificador."""
import sys
import numpy as np
import lab
from lab import *

nom = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else "g1"
P = {}
def pst(e):
    k = e["key"]
    if k not in P:
        P[k] = np.load(f"{lab.CACHE}/stack_{nom}/{k[0]}/{k[1]}_s{k[2]}.npy").astype(np.float64)
    return P[k]

def decode(e, st, cfg):
    prior = np.array(st["prior"])
    eps = cfg.get("eps", 0.1)
    q = e["q"]
    if eps != 0.1:
        q = np.where(q > 0.5, 1 - eps, np.where(np.isclose(q, 1 / 3), 1 / 3, eps / 2))
    em = cfg["a"] * (np.log(np.maximum(pst(e), 1e-6)) - np.log(prior)[None]) + cfg["b"] * np.log(q)
    if cfg.get("d"):   # evidencia solo de balón muerto: las reglas dicen D
        em = em.copy(); em[:, 2] += cfg["d"] * (e["janv1"] == "D")
    if cfg.get("t"):
        em = em + cfg["t"] * (np.log(np.maximum(e["p"], 1e-6)) - np.log(prior)[None])
    m, lq, ls = hmm(st, cfg["s"], cfg.get("sd"))     # sd: escala de duración mínima propia para D (idea M1 de v9m1)
    return per_parts(e, lambda a, b: viterbi(em[a:b], m, lq, ls, np.log(prior)))

if mode == "g1":
    g = [{"a": a, "b": b, "s": s} for a in (0.1, 0.15, 0.25, 0.35, 0.5) for b in (0.0, 0.05) for s in (0.25, 0.5, 1.0)]
elif mode == "g2":   # stack + TCN original
    g = [{"a": a, "b": 0.0, "t": t, "s": s} for a in (0.1, 0.15, 0.25) for t in (0.0, 0.1) for s in (0.25, 0.5)]
elif mode.startswith("b"):   # peso fijo de las reglas en el decodificador (b = mode[1:]/100), eps opcional
    bb = int(mode[1:3]) / 100.0
    ee = 0.02 if mode.endswith("e") else 0.1
    g = [{"a": a, "b": bb, "s": s, "eps": ee} for a in (0.15, 0.25, 0.35) for s in (0.25, 0.5)]
elif mode.startswith("d"):   # bono fijo a D donde las reglas dicen D
    dd = int(mode[1:]) / 100.0
    g = [{"a": a, "b": 0.0, "s": s, "d": dd} for a in (0.15, 0.25, 0.35) for s in (0.25, 0.5)]
elif mode == "g3":   # rejilla final: stack + bono D de las reglas
    g = [{"a": a, "b": 0.0, "s": s, "d": d} for d in (0.0, 0.2, 0.4) for a in (0.15, 0.25, 0.35) for s in (0.25, 0.5)]
elif mode == "g4":   # E14: rejilla g3 ampliada con mezcla con la TCN (t) y duración mínima propia de D (sd)
    g = [{"a": a, "b": 0.0, "s": s, "d": d, "t": t, "sd": sd} for d in (0.0, 0.2) for a in (0.25, 0.35, 0.5) for s in (0.25, 0.5)
         for t in (0.0, 0.1, 0.2) for sd in (None, 0.25, 1.0)]
elif mode == "fix":   # sin selección: una sola configuración para todos los pliegues
    g = [{"a": 0.35, "b": 0.0, "s": 0.5, "d": 0.2}]
if mode == "raw":
    for font, p in SELECCIO[:1] + SELECCIO[-1:]:
        e = entrada(font, p, 0); y = pst(e).argmax(1).astype(np.int8); y[e["period"] == 0] = 3
        print(font, p, "argmax spellF1", spell_f1(e["gt_acc"], y, True), "macro", macro_f1(e["gt_acc"], y), "| TCN macro", macro_f1(e["gt_acc"], np.where(e["period"] > 0, e["p"].argmax(1), 3)))
else:
    avalua_familia(f"stack_{nom}_{mode}", decode, g)
