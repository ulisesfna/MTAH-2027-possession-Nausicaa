"""Agregados del tracking del BAR-ATH que usa la degradación v8 (sin etiquetas; solo se guardan agregados).
   -> target8.json"""
import json
import numpy as np
from common import *
import sig

def centrat(d):
    L, W = d["pitch"]
    b = np.stack([d["bx"].astype(float) - L / 2, W / 2 - d["by"].astype(float)], 1)
    xy = np.stack([d["px"].astype(float) - L / 2, W / 2 - d["py"].astype(float)], 2)
    return b, xy, np.asarray(d["period"]), (L, W)

if __name__ == "__main__":
    b, xy, per, camp = centrat(partit())
    t = {"descriptors": sig.descriptors(b, xy, per, camp)}
    iv, idd, il = sig.celles(b, xy, camp)
    vis = np.isfinite(b[:, 0]) & (per > 0)
    opp = vis.copy(); opp[1:] &= vis[:-1]; opp[0] = False        # balón visible en t y t-1
    g = sig.forats(b, per)
    ini = np.zeros(len(b), bool)
    for s, e in g:
        if e - s >= sig.LLARG: ini[s - 1] = True
    sh = (len(sig.V_BINS) + 1, len(sig.D_BINS) + 1, len(sig.L_BINS) + 1)
    N = np.zeros(sh); S = np.zeros(sh)
    okc = opp & (iv >= 0)
    np.add.at(N, (iv[okc], idd[okc], il[okc]), 1); np.add.at(S, (iv[okc & ini], idd[okc & ini], il[okc & ini]), 1)
    # modelo multiplicativo h = base * fv * fd * fl ajustado por IPF (Poisson) a los marginales
    fv, fd, fl = np.ones(sh[0]), np.ones(sh[1]), np.ones(sh[2]); base = S.sum() / N.sum()
    for _ in range(200):
        E = base * N * fv[:, None, None] * fd[None, :, None] * fl[None, None, :]
        fv *= S.sum((1, 2)) / np.maximum(E.sum((1, 2)), 1e-12)
        E = base * N * fv[:, None, None] * fd[None, :, None] * fl[None, None, :]
        fd *= S.sum((0, 2)) / np.maximum(E.sum((0, 2)), 1e-12)
        E = base * N * fv[:, None, None] * fd[None, :, None] * fl[None, None, :]
        fl *= S.sum((0, 1)) / np.maximum(E.sum((0, 1)), 1e-12)
    t["inici"] = {"V_BINS": sig.V_BINS, "D_BINS": sig.D_BINS, "L_BINS": sig.L_BINS, "base": float(base), "fv": fv.tolist(), "fd": fd.tolist(), "fl": fl.tolist(),
                  "starts_v": S.sum((1, 2)).tolist(), "starts_d": S.sum((0, 2)).tolist(), "starts_l": S.sum((0, 1)).tolist(),
                  "opp_v": N.sum((1, 2)).tolist(), "opp_d": N.sum((0, 2)).tolist(), "opp_l": N.sum((0, 1)).tolist()}
    dl = np.array([e - s for s, e in g if e - s >= sig.LLARG]); dm = np.array([e - s for s, e in g if e - s < sig.LLARG])
    t["durada_llarg_q101"] = np.percentile(dl, np.arange(101)).tolist()
    t["micro"] = {"taxa": float(len(dm) / max(opp.sum(), 1)), "durada_q101": np.percentile(dm, np.arange(101)).tolist() if len(dm) else None, "n": int(len(dm))}
    with open(os.path.join(AQUI, "target8.json"), "w") as fh:
        json.dump(t, fh, indent=1)
    print(json.dumps({k: v for k, v in t.items() if k != "durada_llarg_q101"}, indent=1)[:3500])
