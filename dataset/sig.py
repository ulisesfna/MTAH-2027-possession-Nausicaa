"""Señales del tracking comunes al partido y a los externos (coordenadas centradas, metros, 25 fps)."""
import numpy as np
FPS = 25
V_BINS = (1.0, 3.0, 8.0)        # velocidad del balón, m/s
D_BINS = (1.5, 4.0)             # distancia del jugador más cercano, m
L_BINS = (2.0, 10.0)            # distancia a la línea más cercana (negativa = fuera), m
LLARG = 16                      # un hueco "largo" dura >= 16 frames (0,64 s); por debajo, micro-hueco


def vel(b):
    v = np.full(len(b), np.nan)
    v[1:] = np.hypot(np.diff(b[:, 0]), np.diff(b[:, 1])) * FPS
    return v


def acc(b):
    a = np.full(len(b), np.nan)
    a[1:-1] = np.hypot(b[2:, 0] - 2 * b[1:-1, 0] + b[:-2, 0], b[2:, 1] - 2 * b[1:-1, 1] + b[:-2, 1]) * FPS * FPS
    return a


def d1(b, xy):
    d = np.hypot(xy[..., 0] - b[:, None, 0], xy[..., 1] - b[:, None, 1])
    return np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1)


def linia(b, camp):
    L, W = camp
    return np.minimum(L / 2 - np.abs(b[:, 0]), W / 2 - np.abs(b[:, 1]))


def celles(b, xy, camp):
    """Índices (iv, id, il) por frame; -1 donde no hay balón."""
    v = vel(b)
    v = np.where(np.isfinite(v), v, np.roll(v, -1))  # primer frame de un tramo: la velocidad siguiente
    with np.errstate(invalid="ignore"):
        iv = np.digitize(v, V_BINS); idd = np.digitize(d1(b, xy), D_BINS); il = np.digitize(linia(b, camp), L_BINS)
    ok = np.isfinite(b[:, 0]) & np.isfinite(v)
    return np.where(ok, iv, -1), np.where(ok, idd, -1), np.where(ok, il, -1)


def trams(m):
    e = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def forats(b, per):
    """Huecos de balón [a, e) con balón visible justo antes y después, dentro de la misma parte."""
    absent = ~np.isfinite(b[:, 0]) & (per > 0)
    return [(a, e) for a, e in trams(absent) if a > 0 and e < len(b) and per[a - 1] == per[a] == per[e] and np.isfinite(b[a - 1, 0]) and np.isfinite(b[e, 0])]


def q(x, ps=(5, 10, 25, 50, 75, 90, 95)):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [round(float(t), 4) for t in np.percentile(x, ps)] if len(x) else None


def descriptors(b, xy, per, camp):
    """Agregados sin etiquetas que describen el tracking (para objetivo y para validar)."""
    m = per > 0
    vis = np.isfinite(b[:, 0]) & m
    v, a, dd, li = vel(b), acc(b), d1(b, xy), linia(b, camp)
    g = forats(b, per); gl = [(s, e) for s, e in g if e - s >= LLARG]
    vv = v[vis & np.isfinite(v)]
    pa = np.hypot(xy[2:, :, 0] - 2 * xy[1:-1, :, 0] + xy[:-2, :, 0], xy[2:, :, 1] - 2 * xy[1:-1, :, 1] + xy[:-2, :, 1]) * FPS * FPS
    pv = np.hypot(np.diff(xy[..., 0], axis=0), np.diff(xy[..., 1], axis=0)) * FPS
    vr = [e - s for s, e in trams(vis)]
    aa = a[vis & np.isfinite(a)]
    near = dd < 1.0
    return {
        "visible": float(vis.sum() / m.sum()),
        "bal_vel_q": q(vv), "bal_lent_005": float(np.mean(vv < 0.05)), "bal_lent_03": float(np.mean(vv < 0.3)), "bal_lent_1": float(np.mean(vv < 1.0)),
        "bal_acc_q": q(aa), "bal_acc_gt30": float(np.mean(aa > 30)),
        "bal_fora": float(np.mean(li[vis] < 0)), "d1_q": q(dd[vis]), "d1_lt1": float(np.mean(dd[vis] < 1)),
        "acc15_prop_d1lt1": float(np.mean(a[vis & near & np.isfinite(a)] > 15)), "acc15_prop_d1gt3": float(np.mean(a[vis & (dd > 3) & np.isfinite(a)] > 15)),
        "tram_vis_s_q": q(np.array(vr) / FPS), "forat_s_q": q(np.array([e - s for s, e in g]) / FPS), "n_forats_llargs_per_h": len(gl) / (m.sum() / FPS / 3600),
        "n_micro_per_h": (len(g) - len(gl)) / (m.sum() / FPS / 3600),
        "ini_vel_q": q([v[s - 1] for s, e in gl]), "ini_d1_q": q([dd[s - 1] for s, e in gl]), "ini_lin_q": q([li[s - 1] for s, e in gl]),
        "fi_vel_q": q([v[e + 1] if e + 1 < len(v) else np.nan for s, e in gl]), "fi_d1_q": q([dd[e] for s, e in gl]), "fi_lin_q": q([li[e] for s, e in gl]),
        "despl_q": q([np.hypot(*(b[e] - b[s - 1])) for s, e in gl]),
        "jug_vel_q": q(pv), "jug_acc_q": q(pa), "jug_visibles": float(np.isfinite(xy[m, :, 0]).sum(1).mean()),
    }
