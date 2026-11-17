"""Features del stack de `ulises` (numpy puro; lo usan la inferencia `predict.py` y el entrenamiento `stack.py`).
GENERADO a partir de stack.py: si cambias build() allí, vuelve a generar este fichero (ver test en predict.py --comprueba)."""
import numpy as np


def codis(lab):
    """H/A/D/X -> 0/1/2/3 (igual que evaluation/possession_metrics.label_codes)."""
    lab = np.asarray(lab).astype("<U1")
    out = np.full(len(lab), 3, np.int8)
    for i, c in enumerate("HAD"):
        out[lab == c] = i
    return out


OFFS = (-50, -25, -12, -5, 5, 12, 25, 50)
XN = ["pilota_present", "pilota_vel", "pilota_acc", "pilota_temps_sense_log", "loc_d1", "loc_d2", "loc_relv1",
      "vis_d1", "vis_d2", "vis_relv1", "dist_banda", "dist_fons", "loc_visibles", "vis_visibles"]


def shift(a, k):
    out = np.empty_like(a)
    if k > 0:
        out[:-k] = a[k:]; out[-k:] = a[-1]
    else:
        out[-k:] = a[:k]; out[:-k] = a[0]
    return out


def rmean(a, w):
    c = np.cumsum(np.concatenate(([np.zeros_like(a[0])], a)), axis=0)
    n = len(a); h = w // 2
    lo = np.clip(np.arange(n) - h, 0, n); hi = np.clip(np.arange(n) + h + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def since_change(lab_codes):
    """frames desde el último cambio y hasta el siguiente (cap 250)."""
    n = len(lab_codes)
    ch = np.flatnonzero(lab_codes[1:] != lab_codes[:-1]) + 1
    idx = np.arange(n)
    prev = np.concatenate(([0], ch))[np.searchsorted(ch, idx, side="right")]
    nxt = np.concatenate((ch, [n]))[np.searchsorted(ch, idx, side="right")]
    return np.minimum(idx - prev, 250), np.minimum(nxt - idx, 250)


def desde_hasta(mask, cap=500):
    """frames desde el último True y hasta el siguiente (cap)."""
    n = len(mask); idx = np.arange(n)
    last = np.maximum.accumulate(np.where(mask, idx, -10 ** 6))
    nxt = np.minimum.accumulate(np.where(mask, idx, 10 ** 6)[::-1])[::-1]
    return np.minimum(idx - last, cap), np.minimum(nxt - idx, cap)


GRUPS = ("tcn", "jv", "raw", "touch", "kick")


def build(e, swap=False, fs=GRUPS):
    """Features (n, F). Con swap, los equipos intercambiados (H<->A)."""
    X = e["X"]; noms = e["noms"]
    lp = np.log(np.maximum(e["p"], 1e-6)).astype(np.float32)
    jv = codis(e["janv1"]).astype(np.int8)
    xs = {n: X[:, noms.index(n)] for n in XN}
    if swap:
        lp = lp[:, [1, 0, 2]]
        jv = np.where(jv < 2, 1 - jv, jv).astype(np.int8)
        for a in ("d1", "d2", "relv1", "visibles"):
            xs["loc_" + a], xs["vis_" + a] = xs["vis_" + a], xs["loc_" + a]
    cols, names = [], []
    grup = ["tcn"]
    def add(n, v):
        # "-vel" en la lista de grupos quita la velocidad y aceleración del balón (E19: en el partido el balón muerto
        # visible se mueve y en el banco está parado; el stack lo usaba como atajo para decir "en juego")
        if grup[0] in fs and not ("-vel" in fs and (n in ("pilota_vel", "pilota_acc") or n.startswith("bvel"))):
            names.append(n); cols.append(np.asarray(v, np.float32))
    for i, c in enumerate("HAD"):
        add(f"lp{c}", lp[:, i])
    for k in OFFS:
        s = shift(lp, k)
        for i, c in enumerate("HAD"):
            add(f"lp{c}{k:+d}", s[:, i])
    for w in (25, 75, 151):
        r = rmean(np.exp(lp), w)
        for i, c in enumerate("HAD"):
            add(f"pm{c}{w}", r[:, i])
    grup[0] = "jv"
    oh = np.stack([(jv == i) for i in range(3)], 1).astype(np.float32)
    for i, c in enumerate("HAD"):
        add(f"jv{c}", oh[:, i])
    for k in (-25, -12, 12, 25):
        s = shift(oh, k)
        for i, c in enumerate("HAD"):
            add(f"jv{c}{k:+d}", s[:, i])
    for w in (25, 75):
        r = rmean(oh, w)
        for i, c in enumerate("HAD"):
            add(f"jvm{c}{w}", r[:, i])
    a, b = since_change(jv)
    add("jv_desde", a); add("jv_hasta", b)
    grup[0] = "tcn"
    am = lp.argmax(1)
    a, b = since_change(am)
    add("tcn_desde", a); add("tcn_hasta", b)
    grup[0] = "jv"
    add("clar", e["clar"]); add("phys", e["phys_conf"])
    grup[0] = "raw"
    for n in XN:
        add(n, xs[n])
    add("d1_dif", xs["loc_d1"] - xs["vis_d1"])
    for k in (-12, -5, 5, 12):
        add(f"d1_dif{k:+d}", shift(xs["loc_d1"] - xs["vis_d1"], k))
        add(f"bvel{k:+d}", shift(xs["pilota_vel"], k))
    for w in (13, 51):
        add(f"d1_dif_m{w}", rmean((xs["loc_d1"] - xs["vis_d1"])[:, None], w)[:, 0])
        add(f"loc_d1_min{w}", -rmean(-np.minimum(xs["loc_d1"], 30)[:, None], w)[:, 0])
        add(f"vis_d1_min{w}", -rmean(-np.minimum(xs["vis_d1"], 30)[:, None], w)[:, 0])
    grup[0] = "touch"
    pres = xs["pilota_present"] > 0.5
    for thr in (1.0, 2.0):
        dh = {}
        for T in ("loc", "vis"):
            c = pres & (xs[T + "_d1"] < thr)
            a, b = desde_hasta(c)
            dh[T] = (a, b)
            add(f"{T}_desde_c{thr}", np.log1p(a)); add(f"{T}_hasta_c{thr}", np.log1p(b))
            k = c & (xs["pilota_acc"] > 15.0)
            a, b = desde_hasta(k)
            grup[0] = "kick"
            add(f"{T}_desde_k{thr}", np.log1p(a)); add(f"{T}_hasta_k{thr}", np.log1p(b))
            grup[0] = "touch"
        add(f"ult_c{thr}", np.sign(dh["vis"][0] - dh["loc"][0]))   # +1: el último contacto fue del local
        add(f"seg_c{thr}", np.sign(dh["vis"][1] - dh["loc"][1]))   # +1: el siguiente contacto es del local
    return np.stack(cols, 1), names
