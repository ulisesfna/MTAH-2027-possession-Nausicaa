"""C11 (v8.3): balón muerto visible que se mueve. Post-proceso sobre ficheros ya degradados (cualquier banco).
En el BAR-ATH el balón muerto visible casi nunca está quieto (lo traen al punto de saque, lo recoloca un jugador); en
los bancos está parado, y el stack aprendió "balón visible en movimiento = en juego" (E18/E19 en ulises/REPORT.md).
Mecanismo, sin etiquetas: en cada tramo visible con el balón parado (v < v_parat durante >= t_parat frames), con
probabilidad p_mou, el balón hace una excursión de ida y vuelta (seno) que empieza y acaba en el punto original, a una
velocidad media sorteada (lognormal, mediana v_med); el último segundo y medio queda quieto (colocado para el saque).
    python carry11.py calibra [k=v ...]                 # descriptores de 3 IDSSE (semilla 0) frente al BAR-ATH
    python carry11.py aplica <dir origen> <dir destino> [k=v ...]
"""
import glob, json, os, sys, zlib
import numpy as np
from common import *
import sig
from build_labels import desa_npz
P = {"v_parat": 0.8, "t_parat": 75, "w_s": 25, "t_max": 250, "n_exc": 4, "p_mou": 0.8, "v_med": 2.0, "v_sd": 0.5, "quiet": 37}
with open(os.path.join(AQUI, "target8.json")) as fh:
    T8 = json.load(fh)


def aplica_carry(b, per, rng, P):
    b = b.copy(); v = sig.vel(b); vis = np.isfinite(v); v = np.where(vis, v, 99.0)
    k = np.ones(P["w_s"]) / P["w_s"]; v = np.where(vis, np.convolve(v, k, mode="same"), 99.0)     # velocidad suavizada: el vaivén de v8 no corta el tramo
    n = 0
    for a, e in sig.trams((v < P["v_parat"]) & (per > 0) & np.isfinite(b[:, 0])):
        L = e - a - min(P["quiet"], (e - a) // 3)      # el final queda quieto (balón colocado); en tramos cortos, un tercio
        if e - a < P["t_parat"] or L < 12 or rng.uniform() >= P["p_mou"]:
            continue
        # n_exc excursiones de ida y vuelta de como mucho t_max frames repartidas por el tramo (el balón se mueve a ratos)
        nexc = max(1, min(P["n_exc"], L // P["t_max"] + 1))
        for j in range(nexc):
            T = min(P["t_max"], L // nexc); a0 = a + j * (L // nexc)
            vel = float(np.exp(np.log(P["v_med"]) + P["v_sd"] * rng.normal()))
            A = vel * T / 50.0                  # velocidad media de |d/dt A sin(pi t/T)| = 2A/(T/25)
            ang = rng.uniform(0, 2 * np.pi)
            d = A * np.sin(np.pi * np.arange(T) / T)
            b[a0:a0 + T, 0] += d * np.cos(ang); b[a0:a0 + T, 1] += d * np.sin(ang)
        n += 1
    return b, n


def processa(src, dst, P):
    z = np.load(src, allow_pickle=False); d = {k: z[k] for k in z.files}
    meta = json.loads(str(d["meta"]))
    rng = np.random.default_rng([zlib.crc32(os.path.basename(src).encode()), 11])
    b2, n = aplica_carry(d["ball_xyz"][:, :2].astype(np.float64), d["period"], rng, P)
    d["ball_xyz"] = d["ball_xyz"].copy(); d["ball_xyz"][:, :2] = b2.astype(np.float32)
    info = json.loads(str(d["deg_meta"])) if "deg_meta" in d else {}
    info["c11"] = {"params": P, "trams_moguts": n}; d["deg_meta"] = np.array(json.dumps(info, sort_keys=True))
    if dst:
        desa_npz(dst, d)
    return d, meta["camp_m"], n


if __name__ == "__main__":
    for s in sys.argv[2:] if sys.argv[1] == "calibra" else sys.argv[4:]:
        k, v = s.split("="); P[k] = type(P[k])(float(v))
    if sys.argv[1] == "calibra":
        # objetivo (sin etiquetas del partido): en el BAR-ATH, balón visible donde reglas y TCN dicen D: mediana 1,41 m/s,
        # 24 % por debajo de 0,3. Se compara con los frames D reales (etiqueta externa) del banco, balón visible.
        for nom, arrel in (("v8", V8), ("Jan v3", V3)):
            for tag in ("antes", "C11"):
                vs = []
                for f in sorted(glob.glob(f"{arrel}/idsse/*_s0.npz"))[:3]:
                    d, camp, n = processa(f, None, P) if tag == "C11" else (np.load(f), None, 0)
                    v = sig.vel(d["ball_xyz"][:, :2].astype(float)); g = d["label_acc"].astype("<U1")
                    vs.append(v[(d["period"] > 0) & np.isfinite(v) & (g == "D")])
                v = np.concatenate(vs)
                print(f"{nom:7s} {tag:6s} D visible: vel q10/25/50/75/90 {np.percentile(v, [10, 25, 50, 75, 90]).round(2)}  frac<0.3 {np.mean(v < 0.3):.3f}")
        print("objetivo BAR-ATH (reglas D y TCN D, visible): q [0.1 0.31 1.41 4.16 11.12] frac<0.3 0.244 | params", P)
    else:
        src, dst = sys.argv[2], sys.argv[3]
        for f in sorted(glob.glob(f"{src}/*.npz")):
            _, _, n = processa(f, os.path.join(dst, os.path.basename(f)), P); print(os.path.basename(f), n, flush=True)
