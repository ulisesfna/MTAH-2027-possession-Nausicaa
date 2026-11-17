"""Característiques per frame per a la TCN, idèntiques per al partit i per als externs.

    X, noms = features(estructura)      # estructura de bench/adapter.py

Per equip i sense IDs. Els valors absents es posen a 0, amb un indicador binari, sense imputar
posicions. No fan servir cap camp que només tingui un domini (velocitat de l'ATD, alçada de la
pilota, IDs, Smart Tagging). Vegeu docs/caracteristiques.md.

Memòria cau (fora del repo):
    /workspace/shared/venvs/bench/bin/python bench/features.py --src extern   # features_v7b/<font>/<partit>_s<llavor>.npz
    /workspace/shared/venvs/bench/bin/python bench/features.py --src partit   # features_match/ (protegit)
"""
import argparse
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import adapter  # noqa: E402
import versio  # noqa: E402
from build_labels import desa_npz  # noqa: E402

FPS = adapter.FPS
INICI_PART_S = 10  # segons des del xut inicial de cada part per decidir la direcció d'atac
DADES = versio.DADES
SORTIDA_EXT = versio.FEATS
SORTIDA_PARTIT = versio.LOCAL + "/features_match" if versio.PROPI else "/workspace/shared/features_match"
EQUIPS = (("loc", 0), ("vis", 1))

# indicador que diu quan és vàlida cada característica (per a les estadístiques condicionades)
VALIDESA = {
    "pilota_x": ["pilota_present"], "pilota_y": ["pilota_present"], "pilota_vel": ["pilota_present"],
    "pilota_acc": ["pilota_present"], "dist_banda": ["pilota_present"], "dist_fons": ["pilota_present"],
    "pilota_ultima_x": ["pilota_vista_part"], "pilota_ultima_y": ["pilota_vista_part"],
}
for e, _ in EQUIPS:
    VALIDESA.update({f"{e}_d1": ["pilota_present", f"{e}_te_jugadors"],
                     f"{e}_relv1": ["pilota_present", f"{e}_te_jugadors"],
                     f"{e}_d2": ["pilota_present", f"{e}_te_2"],
                     f"{e}_centre_x": [f"{e}_te_jugadors"], f"{e}_centre_y": [f"{e}_te_jugadors"],
                     f"{e}_dispersio": [f"{e}_te_2"], f"{e}_vel_mitjana": [f"{e}_te_jugadors"]})


def xut_inicial(d, h, a, b):
    """Frame del xut inicial de la part; si no se sap o cau fora del bloc, l'inici del bloc."""
    k = d.get("xuts_inicials", {}).get(int(h), a)
    return k if a <= k <= b else a


def periode_de_joc(d):
    """Com d["period"], però cada part comença al xut inicial: els frames del bloc d'abans del xut
    van a 0, igual que els de fora de les parts als externs."""
    per = np.asarray(d["period"]).copy()
    for h, a, b in d["half_blocks"]:
        per[a:xut_inicial(d, h, a, b)] = 0
    return per


def direccio(d, xc_jug):
    """+1 o -1 per frame: signe que fa que el local ataqui cap a +x. Multiplica x i y (rotació de
    180°, no mirall). Es decideix amb els primers segons des del xut inicial de cada part pels
    centroides dels dos equips (el local defensa el costat on té el centroide)."""
    s = np.ones(d["n_frames"], np.float32)
    info = {}
    eq = d["team_of_track"]
    for h, a, b in d["half_blocks"]:
        k0 = xut_inicial(d, h, a, b)
        fi = min(b + 1, k0 + INICI_PART_S * FPS)
        cx = [np.nanmean(np.where(np.isfinite(xc_jug[k0:fi][:, eq == k]), xc_jug[k0:fi][:, eq == k], np.nan))
              if np.isfinite(xc_jug[k0:fi][:, eq == k]).any() else np.nan for k in (0, 1)]
        if np.isfinite(cx).all():
            sh = -1.0 if cx[0] > cx[1] else 1.0
        else:
            sh = 1.0
        s[a:b + 1] = sh
        info[int(h)] = {"signe": sh, "decidit": bool(np.isfinite(cx).all())}
    return s, info


def temps_des_de(present, periode):
    """Frames des de l'última vegada present dins de la part, i si ja s'ha vist en aquesta part."""
    n = len(present)
    t = np.zeros(n, np.float32)
    vista = np.zeros(n, bool)
    ultim = np.full(n, -1, np.int64)
    for p in sorted(set(periode.tolist()) - {0}):
        idx = np.flatnonzero(periode == p)
        a, b = idx[0], idx[-1] + 1
        pos = np.where(present[a:b], np.arange(a, b), -1)
        u = np.maximum.accumulate(pos)
        ultim[a:b] = u
        vista[a:b] = u >= 0
        t[a:b] = np.where(u >= 0, np.arange(a, b) - u, np.arange(a, b) - a)
    return t, vista, ultim


def features(d):
    n = d["n_frames"]
    L, W = d["pitch"]
    periode = periode_de_joc(d)
    xc_b = d["bx64"] - L / 2
    yc_b = W / 2 - d["by64"]
    xc_j = d["px64"] - L / 2
    yc_j = W / 2 - d["py64"]
    s, info_dir = direccio(d, xc_j)
    pb = np.isfinite(xc_b)
    f = {}
    z = lambda a, m: np.where(m, a, 0.0)

    # pilota
    f["pilota_x"] = z(s * xc_b / (L / 2), pb)
    f["pilota_y"] = z(s * yc_b / (W / 2), pb)
    f["pilota_present"] = pb.astype(float)
    bvx, bvy = d["bvx"].astype(float), d["bvy"].astype(float)
    f["pilota_vel"] = z(np.hypot(bvx, bvy), pb)
    f["pilota_acc"] = z(np.hypot(d["bax"].astype(float), d["bay"].astype(float)), pb)
    t, vista, ultim = temps_des_de(pb, periode)
    f["pilota_temps_sense_log"] = np.log1p(t / FPS)
    u = np.maximum(ultim, 0)
    f["pilota_ultima_x"] = z(s * xc_b[u] / (L / 2), vista)
    f["pilota_ultima_y"] = z(s * yc_b[u] / (W / 2), vista)
    f["pilota_vista_part"] = vista.astype(float)

    # per equip
    eq = d["team_of_track"]
    for e, k in EQUIPS:
        x, y = xc_j[:, eq == k], yc_j[:, eq == k]
        vx, vy = d["pvx"][:, eq == k].astype(float), d["pvy"][:, eq == k].astype(float)
        pr = np.isfinite(x)
        nv = pr.sum(axis=1)
        te1, te2 = nv >= 1, nv >= 2
        f[f"{e}_visibles"] = nv.astype(float)
        dist = np.where(pr, np.hypot(x - xc_b[:, None], y - yc_b[:, None]), np.inf)
        if dist.shape[1] >= 2:
            ordre = np.argsort(dist, axis=1)[:, :2]
            d1 = np.take_along_axis(dist, ordre[:, :1], axis=1)[:, 0]
            d2 = np.take_along_axis(dist, ordre[:, 1:2], axis=1)[:, 0]
        else:
            ordre = np.zeros((n, 1), int)
            d1, d2 = dist[:, 0] if dist.shape[1] else np.full(n, np.inf), np.full(n, np.inf)
        m1, m2 = pb & te1, pb & te2
        f[f"{e}_d1"] = z(np.where(np.isfinite(d1), d1, 0), m1)
        f[f"{e}_d2"] = z(np.where(np.isfinite(d2), d2, 0), m2)
        if dist.shape[1]:
            i1 = ordre[:, 0]
            rvx = np.take_along_axis(vx, i1[:, None], axis=1)[:, 0] - bvx
            rvy = np.take_along_axis(vy, i1[:, None], axis=1)[:, 0] - bvy
            rel = np.hypot(rvx, rvy)
        else:
            rel = np.zeros(n)
        f[f"{e}_relv1"] = z(np.nan_to_num(rel), m1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cx = np.where(te1, np.nansum(np.where(pr, x, 0), axis=1) / np.maximum(nv, 1), 0.0)
            cy = np.where(te1, np.nansum(np.where(pr, y, 0), axis=1) / np.maximum(nv, 1), 0.0)
            disp = np.sqrt(np.nansum(np.where(pr, (x - cx[:, None]) ** 2 + (y - cy[:, None]) ** 2, 0), axis=1)
                           / np.maximum(nv, 1))
        f[f"{e}_centre_x"] = z(s * cx / (L / 2), te1)
        f[f"{e}_centre_y"] = z(s * cy / (W / 2), te1)
        f[f"{e}_dispersio"] = z(disp, te2)
        f[f"{e}_te_jugadors"] = te1.astype(float)
        f[f"{e}_te_2"] = te2.astype(float)

    # context
    f["dist_banda"] = z(W / 2 - np.abs(yc_b), pb)
    f["dist_fons"] = z(L / 2 - np.abs(xc_b), pb)
    f["part_2"] = (periode == 2).astype(float)
    f["en_part"] = (periode > 0).astype(float)
    tp = np.zeros(n)
    for h, a, b in d["half_blocks"]:
        k0 = xut_inicial(d, h, a, b)
        tp[k0:b + 1] = (np.arange(k0, b + 1) - k0) / max(1, b - k0)
    f["temps_part"] = tp

    # velocitat mitjana dels jugadors visibles (v7b); van al final perquè les 32 primeres no canviïn
    vj = np.hypot(d["pvx"].astype(float), d["pvy"].astype(float))
    pj = np.isfinite(xc_j) & np.isfinite(vj)
    for e, k in EQUIPS:
        m = pj[:, eq == k]
        f[f"{e}_vel_mitjana"] = np.where(m.any(1), np.where(m, vj[:, eq == k], 0).sum(1) / np.maximum(m.sum(1), 1), 0.0)
    f["vel_mitjana_tots"] = np.where(pj.any(1), np.where(pj, vj, 0).sum(1) / np.maximum(pj.sum(1), 1), 0.0)
    noms = list(f)
    X = np.stack([np.nan_to_num(f[k]).astype(np.float32) for k in noms], axis=1)
    return X, noms, info_dir


def _extern(path):
    d = adapter.carrega_extern(path)
    X, noms, info = features(d)
    llavor = int(re.search(r"_s(\d+)\.npz$", path).group(1))
    desti = os.path.join(SORTIDA_EXT, d["font"], f"{d['partit']}_s{llavor}.npz")
    g = d["gt"]
    desa_npz(desti, {"X": X, "noms": np.array(noms), "label_acc": g["label_acc"], "label_dur": g["label_dur"],
                     "valid": g["valid"], "period": g["period"],
                     "meta": np.array(json.dumps({"font": d["font"], "partit": d["partit"], "llavor": llavor,
                                                  "split": d["split"], "direccio": info,
                                                  "finestra_sg": d["finestra_sg"], "finestra_vel": d["finestra_vel"],
                                                  "rotacio": "180"}, sort_keys=True))})
    return desti, X.shape, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="all", choices=["all", "extern", "partit"])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--fonts", nargs="+", default=["idsse", "skillcorner", "metrica"])
    a = ap.parse_args()
    if a.src in ("all", "extern"):
        fitxers = sorted(f for f in glob.glob(os.path.join(DADES, "*", "*.npz"))
                         if os.path.basename(os.path.dirname(f)) in a.fonts)
        with Pool(a.jobs) as pool:
            for desti, shape, info in pool.imap(_extern, fitxers):
                print(os.path.relpath(desti, SORTIDA_EXT), shape, {h: v["signe"] for h, v in info.items()}, flush=True)
    if a.src in ("all", "partit"):
        d = adapter.carrega_partit()
        X, noms, info = features(d)
        desa_npz(os.path.join(SORTIDA_PARTIT, f"{d['partit']}.npz"), {"X": X, "noms": np.array(noms)})
        # només forma i decisions agregades, cap valor
        print("partit:", X.shape, "direcció decidida a les dues parts:", all(v["decidit"] for v in info.values()))


if __name__ == "__main__":
    main()
