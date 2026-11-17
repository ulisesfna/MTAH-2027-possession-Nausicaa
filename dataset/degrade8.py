"""Degradación v8 de IDSSE (propia). Parte de los externos sin degradar de Jan (bench_v7) y escribe en ~/mtah_data/bench_v8.

    P=/workspace/shared/venvs/bench/bin/python
    $P degrade8.py calibra            # tabla de descriptores IDSSE v8 (semilla 0) frente al BAR-ATH
    $P degrade8.py genera --seeds 0 1 2

Diferencias con la v3 de Jan (docs/degradacio.md): no usa ninguna etiqueta (ni label_acc ni la reanudación);
C1 suaviza balón y jugadores; C2 oculta el balón parado; C3 los huecos empiezan según velocidad × jugador cercano ×
línea con las tasas del BAR-ATH; C4 acaban en un golpeo. Se mantiene de Jan: relleno de la franja 0,64-5,12 s,
fragmentación de tracks y slots anónimos. SkillCorner y Metrica: enlaces a los ficheros de la v3 (sin cambios).
"""
import argparse, glob, json, os, sys, zlib
import numpy as np
from common import *
import sig
import degrade as dj            # degradador de Jan (funciones de fragmentación y relleno)
import tracking_stats as ts
from build_labels import desa_npz

with open(os.path.join(AQUI, "target8.json")) as fh:
    T8 = json.load(fh)
with open(os.path.join(BASE, "bench", "target_stats.json")) as fh:
    TJ = json.load(fh)

PARAMS = {
    "w_bal": 9, "w_jug": 15,          # C1: ventanas Savitzky-Golay (orden 2)
    "v_parat": 0.3, "t_parat": 25,    # C2: balón por debajo de v_parat durante >= t_parat frames -> oculto
    "p_parat": 1.0, "sigma_w": 0.0, "phi_w": 0.98,   # C2b: fracción de parones que se ocultan; C7: vaivén del balón (m, AR(1))
    "k": 1.0,                         # C3: multiplicador global de la tasa de inicio (para el % visible)
    "v_ini": 1.5, "d_porta": 1.0, "v_porta": 5.0, "max_enrere": 500,  # C2: el hueco empieza cuando un jugador recoge el balón (lento, o cerca y < v_porta)
    "v_cop": 6.0, "d_cop": 1.2, "mira_cop": 5, "busca_cop": 250, "p_cop": 0.7, "v_mou": 1.0,
    "v_lliure": 2.0, "d_lliure": 1.5, "t_lliure": 0,   # C6: el balón reaparece también si rueda libre (>= v_lliure, a >= d_lliure de todos, >= t_lliure frames)   # C4: el hueco acaba en un golpeo (prob. p_cop) o al moverse
    "escala_dur": 1.0,
    # C8/C9 jugadores: fracción de tiempo de cada jugador en caída individual (balón visible / oculto), duración media (s),
    # multiplicadores de la tasa de caídas masivas y de errores de equipo, y fracción de errores que son intercambio de equipo
    "c8": 1, "f_caiguda_vis": 0.028, "f_caiguda_ocult": 0.068, "tau_caiguda": 3.0, "k_massiva": 1.0, "f_massiva_min": 0.45, "k_equip": 1.0, "p_intercanvi": 0.8,
    "extrap": 0, "cop_natural": 0, "dur_parat": 0.0,   # v9 C10: duración de los huecos de parón (× la distribución del partido; 0 = hasta el golpeo)    # v8.1 (SkillCorner): usar jugadores extrapolados; alargar los huecos propios hasta un golpeo
}


def suavitza(a, w):
    if w <= 1:
        return a.copy()
    out = a.copy()
    for c in range(a.shape[1]):
        out[:, c] = adapter.per_fragment(a[:, c].astype(float), w, 0)
    return out


def forats_v8(rng, b, xy, per, camp, P):
    """Máscara de balón oculto (True = oculto). b: balón limpio suavizado (n, 2).
    Modelo: el tracker pierde el balón cuando va lento (C2: frena hasta pararse; C3: tasa por celda del BAR-ATH) y
    lo recupera en un golpeo (C4: balón que sale a >= v_cop de un jugador a < d_cop)."""
    n = len(b)
    v = sig.vel(b); v = np.where(np.isfinite(v), v, 0.0)
    absent = ~np.isfinite(b[:, 0])
    dd = sig.d1(b, xy)
    # golpeo: jugador cerca ahora y balón rápido poco después
    vmax = np.copy(v)
    for k in range(1, P["mira_cop"] + 1):
        vmax[:-k] = np.maximum(vmax[:-k], v[k:])
    cop = (dd < P["d_cop"]) & (vmax >= P["v_cop"]) & ~absent
    # C6: balón libre en movimiento (lejos de todos y rodando/volando de forma sostenida): un tracker por movimiento lo ve
    if P.get("t_lliure", 0) > 0:
        ll = (v >= P["v_lliure"]) & (dd >= P["d_lliure"]) & ~absent
        for a, e in sig.trams(ll):
            if e - a >= P["t_lliure"]:
                cop[a] = True
    fi_part = {p: int(np.flatnonzero(per == p)[-1]) + 1 for p in np.unique(per) if p > 0}

    def espera_cop(j, p):
        """Primer frame >= j con golpeo (con prob. p_cop); si no, o si no hay golpeo en busca_cop frames, el primero
        con el balón en movimiento (>= v_mou)."""
        lim = min(j + P["busca_cop"], fi_part[p])
        k = j
        if rng.uniform() < P["p_cop"]:
            while k < lim and not cop[k]:
                k += 1
            if k < lim:
                return k
        k = j
        while k < fi_part[p] and (absent[k] or v[k] < P["v_mou"]):
            k += 1
        return k
        k = j
        while k < fi_part[p] and (absent[k] or v[k] < P["v_parat"]):
            k += 1
        return k

    out = absent.copy()
    # C4n (fuentes con huecos propios, SkillCorner): el hueco natural largo también se alarga hasta un golpeo
    if P.get("cop_natural", 0):
        for a, e in sig.trams(absent & (per > 0)):
            if e - a >= sig.LLARG and e < fi_part[per[a]]:
                out[e:espera_cop(e, per[a])] = True
    # C2: balón que frena hasta pararse
    for a, e in sig.trams((v < P["v_parat"]) & ~absent & (per > 0)):
        if e - a < P["t_parat"] or rng.uniform() >= P["p_parat"]:   # solo una parte de los parones se pierde
            continue
        s0 = a
        lim0 = max(1, a - P["max_enrere"])
        # hacia atrás mientras el balón va lento o lo lleva un jugador (cerca y sin velocidad de pase)
        while s0 > lim0 and per[s0 - 1] == per[a] and not absent[s0 - 1] and (v[s0 - 1] < P["v_ini"] or (dd[s0 - 1] < P["d_porta"] and v[s0 - 1] < P["v_porta"])):
            s0 -= 1
        fi = espera_cop(e, per[a])
        if P.get("dur_parat", 0) > 0:   # v9 (C10): el parón no queda oculto entero; dura como un hueco largo del partido y el balón reaparece parado
            d = max(sig.LLARG, int(round(P["dur_parat"] * float(dj.mostra_quantils(rng, T8["durada_llarg_q101"])))))
            fi = min(fi, s0 + d)
        out[s0:fi] = True
    # C3 + C4
    iv, idd, il = sig.celles(b, xy, camp)
    I = T8["inici"]
    fv, fd, fl = np.array(I["fv"]), np.array(I["fd"]), np.array(I["fl"])
    h = np.where(iv >= 0, P["k"] * I["base"] * fv[np.maximum(iv, 0)] * fd[np.maximum(idd, 0)] * fl[np.maximum(il, 0)], 0.0)
    hm = T8["micro"]["taxa"]
    u = rng.uniform(size=n); um = rng.uniform(size=n)
    qd, qm = T8["durada_llarg_q101"], T8["micro"]["durada_q101"]
    t, nf = 1, 0
    while t < n - 1:
        if per[t] <= 0 or out[t] or out[t - 1] or per[t + 1] != per[t]:
            t += 1
            continue
        if u[t] < h[t]:
            d = max(sig.LLARG, int(round(P["escala_dur"] * float(dj.mostra_quantils(rng, qd)))))
            fi = min(t + 1 + d, fi_part[per[t]])
            fi = max(fi, espera_cop(fi, per[t])) if fi < fi_part[per[t]] else fi
            out[t + 1:fi] = True
            nf += 1
            t = fi
        elif um[t] < hm:
            d = max(1, int(round(float(dj.mostra_quantils(rng, qm)))))
            out[t + 1:min(t + 1 + d, fi_part[per[t]])] = True
            t += 1 + d
        else:
            t += 1
    return out, nf


with open(os.path.join(AQUI, "target8_jug.json")) as fh:
    TJG = json.load(fh)


def mostra_q(rng, punts):
    """Muestra por interpolación log-lineal de una función de cuantiles dada por puntos (percentil, valor)."""
    ps, vs = zip(*punts)
    return float(np.exp(np.interp(rng.uniform(0, 100), ps, np.log(vs))))


def jugadors_v8(rng, xy, equip, per, ocult, P):
    """C8: caídas de jugadores ligadas a que el balón no se vea (individuales y masivas).
    C9: detecciones asignadas al equipo contrario y detecciones de más. xy (n, S, 2) con todos presentes (salvo
    los cortes de fragmentación). Devuelve (xy, equip) con columnas nuevas para los intercambios."""
    n, S, _ = xy.shape
    xy = xy.copy()
    m = per > 0
    idx = np.flatnonzero(m)
    # C8a: caídas individuales, con tasa distinta según se vea o no el balón
    tau = P["tau_caiguda"] * sig.FPS
    lam = np.where(ocult, P["f_caiguda_ocult"], P["f_caiguda_vis"]) / tau
    for s_ in range(S):
        u = rng.uniform(size=n)
        t = 0
        ini = np.flatnonzero((u < lam) & m)
        for a in ini:
            if a < t:
                continue
            d = int(rng.geometric(1.0 / tau))
            xy[a:a + d, s_] = np.nan
            t = a + d
    # C8b: caídas masivas (la cámara deja de ver el campo) solo con el balón oculto
    lt14, lt8 = TJG["lt14"], TJG["lt8"]
    punts = [(0, 0.2), (25, lt14["dur_q"][0]), (50, lt14["dur_q"][1]), (75, lt14["dur_q"][2]), (90, lt14["dur_q"][3]), (100, 1.3 * lt14["dur_q"][3])]
    n_ev = rng.poisson(P["k_massiva"] * lt14["ratxes_h"] * m.sum() / 90000.0)
    gaps = [(a, e) for a, e in sig.trams(ocult & m) if e - a >= sig.LLARG]
    pes = np.array([e - a for a, e in gaps], float)
    for _ in range(n_ev):
        if not len(gaps):
            break
        ga, ge = gaps[int(rng.choice(len(gaps), p=pes / pes.sum()))]     # huecos largos, más probable en los más largos
        a = int(ga + rng.uniform(0, 0.3) * (ge - ga))
        d = int(mostra_q(rng, punts) * sig.FPS)
        e = a
        while e < min(a + d, n) and per[e] == per[a] and (ocult[e] or e - a < 12):   # la caída no sobrevive mucho a la reaparición del balón
            e += 1
        f = rng.uniform(P["f_massiva_min"], 1.0)
        quins = rng.uniform(size=S) < f
        xy[a:e, quins] = np.nan
    # C9: equipo erróneo (80 %) o detección de más (20 %)
    e12 = TJG["equip12"]
    punts = [(0, 0.2), (25, e12["dur_q"][0]), (50, e12["dur_q"][1]), (75, e12["dur_q"][2]), (90, e12["dur_q"][3]), (100, 2 * e12["dur_q"][3])]
    n_ev = rng.poisson(P["k_equip"] * e12["ratxes_h"] * m.sum() / 90000.0)
    extra, extra_eq = [], []
    for _ in range(n_ev):
        a = int(idx[rng.integers(len(idx))]); d = max(2, int(mostra_q(rng, punts) * sig.FPS)); e = min(a + d, n)
        pres = np.flatnonzero(np.isfinite(xy[a, :, 0]))
        if not len(pres):
            continue
        s_ = int(pres[rng.integers(len(pres))])
        col = np.full((n, 2), np.nan, np.float32)
        seg = xy[a:e, s_].copy()
        if rng.uniform() < P["p_intercanvi"]:
            col[a:e] = seg; xy[a:e, s_] = np.nan; extra_eq.append(1 - int(equip[s_]))
        else:
            col[a:e] = seg + rng.normal(0, 1.0, size=2); extra_eq.append(int(rng.integers(2)))
        extra.append(col)
    if extra:
        xy = np.concatenate([xy, np.stack(extra, 1)], 1); equip = np.concatenate([equip, np.array(extra_eq, np.int8)])
    return xy.astype(np.float32), equip.astype(np.int8)


def degrada8(path, seed, P, out_dir=None):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    font, partit = meta["font"], meta["partit"]
    rng = np.random.default_rng([seed, zlib.crc32(f"{font}/{partit}".encode()), 8])
    per = z["period"]; camp = meta["camp_m"]
    bola = z["ball_xyz"].copy().astype(np.float64); xy = z["players_xy"].copy().astype(np.float64); equip = z["players_team"]
    bola[~z["ball_detected"]] = np.nan
    if not P.get("extrap", 0):      # extrap=1 (SkillCorner v8.1): las posiciones extrapoladas del proveedor cuentan como vistas
        xy[~z["players_detected"]] = np.nan
    mask = per > 0
    vis_propi = float(np.isfinite(xy[mask, :, 0]).sum(1).mean()) / 2
    # C1
    bola[:, :2] = suavitza(bola[:, :2], P["w_bal"])
    xys = xy.copy()
    for c in range(2):
        xys[:, :, c] = np.stack([adapter.per_fragment(xy[:, s, c], P["w_jug"], 0) for s in range(xy.shape[1])], 1) if P["w_jug"] > 1 else xy[:, :, c]
    # C2-C4
    out, nf = forats_v8(rng, bola[:, :2], xys, per, camp, P)
    if P["sigma_w"] > 0:   # C7: vaivén lento del tracker (proceso AR(1) en posición), para que el balón parado no tenga velocidad 0
        w = rng.normal(size=(len(bola), 2)) * P["sigma_w"] * np.sqrt(1 - P["phi_w"] ** 2)
        e = np.zeros_like(w)
        for t in range(1, len(w)):
            e[t] = P["phi_w"] * e[t - 1] + w[t]
        bola[:, :2] += suavitza(e, 11)
    bola[out] = np.nan
    # C5 (Jan): relleno de la franja vacía, fragmentación y slots anónimos
    franja = dj.franja_buida(TJ)
    bola, n_omp = dj.omple_franja(bola, per, franja)
    if P["c8"]:
        xy2, equip2, n_frag = dj.fragmenta(rng, xys.astype(np.float32), equip, per, TJ, 1.0)   # solo cortes y slots anónimos
        xy2, equip2 = jugadors_v8(rng, xy2, equip2, per, ~np.isfinite(bola[:, 0]), P)
    else:
        p = min(1.0, TJ["jugadors"]["visibles_mitjana"] / vis_propi)
        xy2, equip2, n_frag = dj.fragmenta(rng, xys.astype(np.float32), equip, per, TJ, p)
    info = {"versio": "v8", "llavor": seed, "font": font, "partit": partit, "params": P, "forats": nf, "forats_omplerts": n_omp, "fragments": n_frag}
    d = {k: z[k] for k in dj.CLAUS_ETIQUETA}
    d.update({"ball_xyz": bola.astype(np.float32), "players_xy": xy2, "players_team": equip2, "meta": z["meta"], "deg_meta": np.array(json.dumps(info, sort_keys=True))})
    if out_dir:
        desa_npz(os.path.join(out_dir, font, f"{partit}_s{seed}.npz"), d)
    return d, camp


CLAUS = ["visible", "bal_lent_005", "bal_lent_03", "bal_lent_1", "bal_acc_gt30", "bal_fora", "d1_lt1", "acc15_prop_d1lt1", "acc15_prop_d1gt3", "n_forats_llargs_per_h", "n_micro_per_h", "jug_visibles"]
CLAUS_Q = ["bal_vel_q", "bal_acc_q", "d1_q", "tram_vis_s_q", "forat_s_q", "ini_vel_q", "ini_d1_q", "ini_lin_q", "fi_vel_q", "fi_d1_q", "fi_lin_q", "despl_q", "jug_vel_q", "jug_acc_q"]


def taula(ds, noms):
    """ds: lista de listas de descriptores (una por columna, promediando partidos)."""
    mitj = lambda L, k: np.mean([x[k] for x in L], axis=0)
    print(f"{'':22s}" + "".join(f"{n:>26s}" for n in noms))
    for k in CLAUS:
        print(f"{k:22s}" + "".join(f"{mitj(L, k):26.4f}" for L in ds))
    for k in CLAUS_Q:
        print(f"{k + ' p10..p90':22s}" + "".join("{:>26s}".format("/".join(f"{x:.1f}" for x in mitj(L, k)[[1, 2, 3, 4, 5]])) for L in ds))


def desc_npz(d, camp):
    return sig.descriptors(d["ball_xyz"][:, :2].astype(float), d["players_xy"].astype(float), d["period"], camp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ordre", choices=["calibra", "genera"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--set", nargs="*", default=[], help="param=valor")
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--font", default="idsse")
    ap.add_argument("--des", type=int, default=0, help="índice del primer partido (para repartir en dos procesos)")
    ap.add_argument("--params", help="json de parámetros de partida (p. ej. params_v8.json)")
    a = ap.parse_args()
    P = dict(PARAMS)
    if a.params:
        with open(a.params) as fh:
            P.update(json.load(fh))
    for s in a.set:
        k, v = s.split("="); P[k] = type(PARAMS[k])(float(v))
    fitxers = sorted(glob.glob(f"{NET}/{a.font}/*.npz"))[a.des:a.n]
    if a.ordre == "calibra":
        v8 = [desc_npz(*degrada8(f, 0, P)) for f in fitxers]
        v3 = []
        for f in fitxers:
            z = np.load(f"{V3}/{a.font}/{os.path.basename(f)[:-4]}_s0.npz"); v3.append(sig.descriptors(z["ball_xyz"][:, :2].astype(float), z["players_xy"].astype(float), z["period"], json.loads(str(z["meta"]))["camp_m"]))
        print(json.dumps(P))
        taula([[T8["descriptors"]], v8, v3], ["BAR-ATH", a.font + " v8", a.font + " v3 (Jan)"])
    else:
        for f in fitxers:
            for s in a.seeds:
                degrada8(f, s, P, V8); print(os.path.basename(f), s, flush=True)
        for font in [x for x in ("skillcorner", "metrica") if x != a.font and a.font == "idsse"]:
            os.makedirs(f"{V8}/{font}", exist_ok=True)
            for g in glob.glob(f"{V3}/{font}/*.npz"):
                dst = f"{V8}/{font}/{os.path.basename(g)}"
                if not os.path.lexists(dst):
                    os.symlink(os.path.realpath(g), dst)
        with open(os.path.join(V8, f"params_{a.font}.json"), "w") as fh:
            json.dump(P, fh, indent=1)
