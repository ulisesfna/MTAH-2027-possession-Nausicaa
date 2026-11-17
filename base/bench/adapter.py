"""Mateixa estructura en memòria per al partit proporcionat i per als partits externs degradats.

Els models (starter/baseline.py, starter/principled_v1.py) i bench/features.py treballen sobre arrays
en metres des de la cantonada superior esquerra (x al llarg, y cap avall), un equip per slot i les
dues parts. Aquí es construeixen igual als dos dominis:

    partit = carrega_partit()                       # ATD tactical, carregadors del repo
    extern = carrega_extern(path_npz)               # /workspace/shared/bench_v7_deg/...

Camps: n_frames, pitch (L, W), px/py (n, S) float32, px64/py64/bx64/by64 en float64 (el baseline
compara distàncies en doble precisió), bx/by (n,) float32, pvx/pvy (n, S) i bvx/bvy/bax/bay (n,)
velocitats i acceleracions en m/s i m/s², mean_speed (n,) float32, team_of_track (S,) int8,
half_blocks [(1, inici, fi), (2, inici, fi)] amb fi inclòs, kickoff_2_frame i xuts_inicials
{part: frame del xut inicial}.

Tot es calcula igual als dos dominis, dins de cada fragment. Amb suavitzat, les posicions passen
per un Savitzky-Golay (ordre 2, finestra FINESTRA) i les velocitats i acceleracions són derivades
del mateix filtre. Sense suavitzat, les posicions queden tal com són i les derivades surten d'un
Savitzky-Golay de finestra FINESTRA_VEL. Al partit no es fa servir la velocitat de l'ATD. Quina de
les dues és la de per defecte (SUAVITZA) ho va decidir l'A/B del banc (docs/degradacio.md).
"""
import functools
import json
import math
import os
import re
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
ARREL = os.path.dirname(AQUI)
sys.path.insert(0, ARREL)
sys.path.insert(0, AQUI)
import tracking_stats as ts  # noqa: E402

FPS = 25
ORDRE = 2
FINESTRA = 25      # posicions, quan se suavitzen (bench/tria_finestra.py)
FINESTRA_VEL = 11  # derivades, quan les posicions no se suavitzen
SUAVITZA = False   # per defecte, decidit per l'A/B de docs/degradacio.md


# ---------------------------------------------------------------- Savitzky-Golay per fragments

def _fila(p, t, d):
    """Coeficients que donen la derivada d-èsima del polinomi ajustat, avaluada a l'offset t."""
    r = np.zeros(p.shape[1])
    for j in range(d, p.shape[0]):
        r += math.factorial(j) / math.factorial(j - d) * (t ** (j - d)) * p[j]
    return r


@functools.lru_cache(maxsize=None)
def _files(ww, o, d):
    h = ww // 2
    p = np.linalg.pinv(np.vander(np.arange(ww) - h, o + 1, increasing=True))
    return (_fila(p, 0, d)[::-1].copy(), np.stack([_fila(p, i - h, d) for i in range(h)]),
            np.stack([_fila(p, i + 1, d) for i in range(h)]))


def sg(x, w, d=0):
    """Savitzky-Golay d'una sèrie sense NaN; a les vores avalua el polinomi de la primera i
    l'última finestra. Derivades en unitats per segon."""
    n = len(x)
    x = x.astype(np.float64)
    ww = min(w, n if n % 2 else n - 1)
    if ww < 3:
        if d == 0:
            return x.copy()
        return (np.gradient(x) * FPS if n > 1 else np.zeros(n)) if d == 1 else np.zeros(n)
    h = ww // 2
    centre, inici, final = _files(ww, min(ORDRE, ww - 1), d)
    out = np.empty(n)
    out[h:n - h] = np.convolve(x, centre, mode="valid")
    out[:h] = inici @ x[:ww]
    out[n - h:] = final @ x[n - ww:]
    return out * (FPS ** d)


def per_fragment(x, w, d=0):
    out = np.full(len(x), np.nan)
    for a, b in ts.trams(np.isfinite(x)):
        out[a:b] = sg(x[a:b], w, d)
    return out


def _columnes(a, w, d):
    return np.stack([per_fragment(a[:, s], w, d) for s in range(a.shape[1])], axis=1)


def cinematica(d, w=FINESTRA, suavitza=None):
    """Suavitza (o no) les posicions i hi afegeix velocitats, acceleració de la pilota i mean_speed."""
    suavitza = SUAVITZA if suavitza is None else suavitza
    if not suavitza:
        w = FINESTRA_VEL
    else:
        d["px"] = _columnes(d["px"], w, 0).astype(np.float32)
        d["py"] = _columnes(d["py"], w, 0).astype(np.float32)
        d["bx"] = per_fragment(d["bx"], w, 0).astype(np.float32)
        d["by"] = per_fragment(d["by"], w, 0).astype(np.float32)
    for k in ("px", "py", "bx", "by"):
        d[k + "64"] = d[k].astype(np.float64)
    d["pvx"] = _columnes(d["px"], w, 1).astype(np.float32)
    d["pvy"] = _columnes(d["py"], w, 1).astype(np.float32)
    d["bvx"] = per_fragment(d["bx"], w, 1).astype(np.float32)
    d["bvy"] = per_fragment(d["by"], w, 1).astype(np.float32)
    d["bax"] = per_fragment(d["bx"], w, 2).astype(np.float32)
    d["bay"] = per_fragment(d["by"], w, 2).astype(np.float32)
    v = np.hypot(d["pvx"], d["pvy"])
    n = np.isfinite(v).sum(axis=1)
    with np.errstate(invalid="ignore"):
        d["mean_speed"] = np.where(n > 0, np.nansum(v, axis=1) / np.maximum(n, 1), 0.0).astype(np.float32)
    d["finestra_sg"] = w if suavitza else None
    d["finestra_vel"] = w
    return d


# ---------------------------------------------------------------- càrrega

def blocs_de_periode(periode):
    blocs = []
    for p in (1, 2):
        idx = np.flatnonzero(periode == p)
        if idx.size:
            blocs.append((p, int(idx[0]), int(idx[-1])))
    return blocs


def carrega_extern(path, smooth=None, finestra=FINESTRA):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    L, W = meta["camp_m"]
    xy = z["players_xy"]
    b = z["ball_xyz"]
    periode = z["period"]
    blocs = blocs_de_periode(periode)
    m = re.search(r"_s(\d+)\.npz$", path)
    d = {"domini": "extern", "font": meta["font"], "partit": meta["partit"], "split": meta["split"],
         "llavor": int(m.group(1)) if m else None,
         "n_frames": int(len(periode)), "pitch": (float(L), float(W)),
         "px": (xy[..., 0] + L / 2).astype(np.float32), "py": (W / 2 - xy[..., 1]).astype(np.float32),
         "bx": (b[:, 0] + L / 2).astype(np.float32), "by": (W / 2 - b[:, 1]).astype(np.float32),
         "team_of_track": z["players_team"].astype(np.int8), "period": periode,
         "half_blocks": blocs, "kickoff_2_frame": blocs[1][1] if len(blocs) > 1 else blocs[0][2] + 1,
         "xuts_inicials": {int(h): int(x) for h, x, _ in blocs},  # als externs, cada part comença al xut
         "gt": {k: z[k] for k in ("label_acc", "label_dur", "valid", "period")}}
    return cinematica(d, finestra, smooth)


def carrega_partit(smooth=None, finestra=FINESTRA):
    import games
    from starter import principled_v1 as pv
    meta, raw = games.require(["atd_meta", "atd_raw"])
    n = games.source_frames("tactical")
    pitch = games.GAMES[games.GAME]["pitch"]
    team = pv.get_team_mapping(meta)
    px, py, _, bx, by = pv.parse_atd(raw, n, len(team), pitch)  # la velocitat de l'ATD no es fa servir
    k1 = games.GAMES[games.GAME]["kickoff"]
    k2 = games.GAMES[games.GAME]["kickoff_2nd"]
    xut = lambda k: (k["minute"] * 60 + k["second"]) * FPS + k["frame"]
    periode = np.zeros(n, np.int8)
    for h, a, b in games.half_video_blocks(games.GAME):
        periode[a:b + 1] = h
    d = {"domini": "partit", "font": "partit", "partit": games.GAME, "split": "partit",
         "n_frames": n, "pitch": tuple(pitch), "px": px, "py": py, "bx": bx, "by": by,
         "team_of_track": team, "period": periode, "half_blocks": games.half_video_blocks(games.GAME),
         "kickoff_2_frame": xut(k2),
         # els blocs de vídeo poden començar abans del xut inicial (a la 2a part, gairebé 2 min)
         "xuts_inicials": {1: xut(k1), 2: xut(k2)},
         "smart_tagging": games.asset_paths()["smart_tagging"]}
    return cinematica(d, finestra, smooth)
