"""E5: modelo apilado (LightGBM) sobre las probabilidades fuera de pliegue de la TCN v7d + reglas + features locales.

    .venv/bin/python stack.py <nombre> <alpha>      # alpha: peso por frame ∝ 1/len(spell)^alpha en H/A

Entrena solo con IDSSE (las probs TCN de SkillCorner no son fuera de pliegue). Un modelo por IDSSE
dejado fuera (predice ese partido) y uno con los 7 (predice Metrica). Aumentación por intercambio de
equipos y promedio simétrico al predecir. Guarda probs en cache/stack_<nombre>/<font>/<partit>_s<ll>.npy.
"""
import os, sys, time
import numpy as np
import lightgbm as lgb
import lab
from lab import *

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
    X = feats(e); noms = e["noms"]
    lp = np.log(np.maximum(e["p"], 1e-6)).astype(np.float32)
    jv = pm.label_codes(e["janv1"]).astype(np.int8)
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


def pesos(g, alpha):
    w = np.ones(len(g), np.float32)
    if alpha > 0:
        runs = pm.valid_runs(g)
        for a, b, k in lab._spells(g, runs):
            w[a:b] = (250.0 / max(b - a, 25)) ** alpha
    return w


def main():
    nom, alpha = sys.argv[1], float(sys.argv[2])
    fs = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else GRUPS
    obj = sys.argv[4] if len(sys.argv) > 4 else "gt_acc"   # etiqueta de entrenamiento: gt_acc | gt_dur | gt_prov
    global build
    _b = build
    build = lambda e, sw=False: _b(e, sw, fs)
    pas = int(os.environ.get("ULISES_PAS", "4"))
    # v10: ULISES_LLAVORS_TR = semillas de entrenamiento y predicción (p. ej. 0,1,2,3,4,5; por defecto las de lab);
    # ULISES_SC=1 añade los 20 SkillCorner al entrenamiento (necesita sus probabilidades TCN fuera de pliegue).
    TR = tuple(int(x) for x in os.environ.get("ULISES_LLAVORS_TR", ",".join(map(str, LLAVORS))).split(","))
    SC = lab.partits("skillcorner") if os.environ.get("ULISES_SC", "0") == "1" else []
    PROVA = os.environ.get("ULISES_PROVA", "0") == "1"      # prueba rápida del código: 2 IDSSE, 2 SkillCorner, 10 rondas
    if PROVA:
        globals()["IDSSE"] = IDSSE[:2]; globals()["METRICA"] = METRICA[:1]; SC = SC[:2]
    out = os.path.join(lab.CACHE, f"stack_{nom}")
    t0 = time.time()
    tr = {}
    for font, p in [("idsse", q) for q in IDSSE] + [("skillcorner", q) for q in SC]:
        Xs, ys, ws = [], [], []
        for ll in TR:
            e = entrada(font, p, ll); g = e[obj]
            idx = np.flatnonzero(g != 3)[ll::pas]
            w = pesos(g, alpha)
            for sw in (False, True):
                F, names = build(e, sw)
                y = g[idx].astype(np.int64)
                if sw:
                    y = np.where(y < 2, 1 - y, y)
                Xs.append(F[idx]); ys.append(y); ws.append(w[idx])
            e.pop("X", None)
            if font == "skillcorner":
                lab._mem.pop(e["key"], None)    # no guardar SkillCorner en memoria
        tr[p] = (np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws))
        print("feats", p, tr[p][0].shape, f"{time.time() - t0:.0f}s", flush=True)
    par = dict(objective="multiclass", num_class=3, learning_rate=0.08, num_leaves=31, min_data_in_leaf=200,
               feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0, num_threads=int(os.environ.get("ULISES_FILS", "2")), verbose=-1, seed=0,
               deterministic=True, force_row_wise=True)

    def fit(ps):
        ps = list(ps) + SC
        X = np.concatenate([tr[p][0] for p in ps]); y = np.concatenate([tr[p][1] for p in ps]); w = np.concatenate([tr[p][2] for p in ps])
        return lgb.train(par, lgb.Dataset(X, y, weight=w, feature_name=names), num_boost_round=10 if PROVA else 250)

    def predict(model, font, p):
        for ll in TR:
            e = entrada(font, p, ll)
            a = model.predict(build(e, False)[0]); b = model.predict(build(e, True)[0])[:, [1, 0, 2]]
            e.pop("X", None)
            os.makedirs(f"{out}/{font}", exist_ok=True)
            np.save(f"{out}/{font}/{p}_s{ll}.npy", ((a + b) / 2).astype(np.float32))

    for p in IDSSE:
        m = fit([q for q in IDSSE if q != p]); predict(m, "idsse", p)
        m.save_model(f"{out}/model_{p}.txt")      # v11: se guardan los pliegues para poder evaluarlos en otro banco
        print("plec", p, f"{time.time() - t0:.0f}s", flush=True)
    m = fit(IDSSE)
    for p in METRICA:
        predict(m, "metrica", p)
    m.save_model(f"{out}/model_tots.txt")
    imp = sorted(zip(m.feature_importance("gain"), names), reverse=True)[:25]
    print("importancia:", [(n, int(g)) for g, n in imp])


if __name__ == "__main__":
    main()
