"""Calibración sin etiquetas de la degradación v8: búsqueda por coordenadas que minimiza la distancia entre los
descriptores del tracking de IDSSE v8 (semilla 0, 3 partidos) y los del BAR-ATH (target8.json). -> calib8.json / .log"""
import json, sys, glob
import numpy as np
from common import *
import degrade8 as d8

ESC = ["visible", "bal_lent_005", "bal_lent_03", "bal_lent_1", "bal_acc_gt30", "bal_fora", "d1_lt1", "acc15_prop_d1lt1", "n_forats_llargs_per_h"]
VEC = ["bal_vel_q", "bal_acc_q", "tram_vis_s_q", "forat_s_q", "ini_vel_q", "ini_d1_q", "ini_lin_q", "fi_vel_q", "fi_d1_q", "fi_lin_q", "despl_q"]
EPS = {"n_forats_llargs_per_h": 5.0}


def distancia(desc, tgt=d8.T8["descriptors"], detall=False):
    out = {}
    for k in ESC:
        a, b = float(np.mean([x[k] for x in desc])), tgt[k]
        out[k] = abs(a - b) / (abs(a) + abs(b) + EPS.get(k, 0.01))
    for k in VEC:
        a = np.mean([x[k] for x in desc], axis=0)[1:6]; b = np.array(tgt[k])[1:6]
        out[k] = float(np.mean(np.abs(a - b) / (np.abs(a) + np.abs(b) + 0.5)))
    D = float(np.mean(list(out.values())))
    return (D, out) if detall else D


SONDES = ["pctD_regles", "spells_h_regles", "D_vis_regles", "viu_novis_regles", "pctD_baseline", "viu_novis_baseline", "acord_regles_baseline"]
EPS_S = {"spells_h_regles": 10.0}


def sondes_regles(d_npz):
    """Estadísticos de las reglas janv1 y del baseline (funciones deterministas del tracking, sin etiquetas)."""
    import tempfile, tune_decoder as td, harness as hz
    import numpy as np
    from build_labels import desa_npz
    tmp = os.path.join("/dev/shm", f"calib8_{os.getpid()}_s0.npz")
    desa_npz(tmp, d_npz)
    d = adapter.carrega_extern(tmp); os.remove(tmp)
    C = {"H": 0, "A": 1, "D": 2, "X": 3}; cod = lambda lab: np.vectorize(C.get)(np.asarray(lab).astype("<U1"))
    per = np.asarray(d["period"]); m = per > 0; vis = np.isfinite(d["bx"]) & m
    jv = cod(td.janv1_amb_control(d)[0]); bs = cod(hz.pred_baseline_smooth50(d))
    n = 0
    for p in (1, 2):
        w = jv[per == p]; ch = np.flatnonzero(w[1:] != w[:-1]) + 1; n += int(np.sum(w[np.concatenate(([0], ch))] < 2))
    return {"pctD_regles": float(np.mean(jv[m] == 2)), "spells_h_regles": n / (m.sum() / 25 / 3600), "D_vis_regles": float(np.sum((jv == 2) & vis) / m.sum()),
            "viu_novis_regles": float(np.sum((jv < 2) & m & ~vis) / m.sum()), "pctD_baseline": float(np.mean(bs[m] == 2)),
            "viu_novis_baseline": float(np.sum((bs < 2) & m & ~vis) / m.sum()), "acord_regles_baseline": float(np.mean(jv[m] == bs[m]))}


def distancia2(desc, sond, pes_sondes=float(os.environ.get("PES_SONDES", "2.0"))):
    """Distancia de descriptores + estadísticos de las sondas de reglas (objetivo: probes_partit.json)."""
    D, out = distancia(desc, detall=True)
    tgt = json.load(open(os.path.join(AQUI, "probes_partit.json")))["BAR-ATH"]
    o2 = {}
    for k in SONDES:
        a, b = float(np.mean([x[k] for x in sond])), tgt[k]
        o2[k] = abs(a - b) / (abs(a) + abs(b) + EPS_S.get(k, 0.01))
    tot = (sum(out.values()) + pes_sondes * sum(o2.values())) / (len(out) + pes_sondes * len(o2))
    return float(tot), {**out, **o2}


if __name__ == "__main__":
    fitxers = sorted(glob.glob(f"{NET}/idsse/*.npz"))[:3]
    MODE = sys.argv[1] if len(sys.argv) > 1 else ""
    cache = {}
    log = open(os.path.join(AQUI, "calib8.log"), "a")
    def f(P):
        key = json.dumps(P, sort_keys=True)
        if key not in cache:
            if MODE == "sondes":
                dd = [d8.degrada8(x, 0, P) for x in fitxers]
                cache[key] = distancia2([d8.desc_npz(*t) for t in dd], [sondes_regles(t[0]) for t in dd])[0]
            else:
                cache[key] = distancia([d8.desc_npz(*d8.degrada8(x, 0, P)) for x in fitxers])
            log.write(f"{cache[key]:.4f} {key}\n"); log.flush()
        return cache[key]
    P = dict(d8.PARAMS); P.update({"w_bal": 25, "v_parat": 0.1, "t_parat": 12, "k": 0.3, "escala_dur": 0.6, "p_cop": 0.7})
    if len(sys.argv) > 1 and sys.argv[1] == "amplia":   # segunda búsqueda: parte del resultado anterior y amplía los bordes
        P = json.load(open(os.path.join(AQUI, "calib8.json")))["params"]
    GRID = {"k": [0.0, 0.15, 0.3, 0.6, 1.0], "p_cop": [0.4, 0.7, 1.0], "d_porta": [0.0, 0.6, 1.0, 1.5], "v_porta": [3.0, 5.0, 8.0], "v_ini": [0.1, 0.8, 1.5, 3.0],
            "t_parat": [12, 25, 50, 100], "v_parat": [0.05, 0.1, 0.3], "escala_dur": [0.3, 0.6, 1.0], "w_bal": [15, 25, 35], "v_cop": [4.0, 6.0, 9.0], "d_cop": [0.8, 1.2, 2.0],
            "busca_cop": [125, 250, 500], "max_enrere": [125, 500]}
    if MODE == "sondes":   # tercera búsqueda: añade al objetivo el comportamiento de las reglas y del baseline
        P = json.load(open(os.path.join(AQUI, "calib8.json")))["params"]
        P = {**d8.PARAMS, **P}
        GRID = {"sigma_w": [0.0, 0.03, 0.06, 0.1, 0.15], "p_parat": [0.4, 0.6, 0.8, 1.0], "p_cop": [0.0, 0.3, 0.5, 0.7], "k": [0.3, 0.6, 1.0, 1.4], "busca_cop": [25, 60, 125], "v_mou": [0.5, 1.0, 2.0, 3.0], "t_parat": [75, 150, 300, 600], "escala_dur": [0.2, 0.3, 0.45, 0.6],
                "v_cop": [5.0, 7.5, 10.0], "w_bal": [35, 55], "v_parat": [0.1, 0.3]}
    if len(sys.argv) > 1 and sys.argv[1] == "amplia":
        GRID = {"w_bal": [29, 35, 45, 55], "w_jug": [9, 15, 25], "k": [0.8, 1.0, 1.3, 1.7], "t_parat": [75, 100, 150, 250], "busca_cop": [50, 90, 125], "v_cop": [7.5, 9.0, 11.0],
                "escala_dur": [0.45, 0.6, 0.8], "p_cop": [0.55, 0.7, 0.85], "d_cop": [1.0, 1.2, 1.5], "v_mou": [0.5, 1.0, 2.0], "mira_cop": [3, 5, 10]}
    best = f(P)
    for ronda in range(2):
        for k, vals in GRID.items():
            for v in vals:
                Q = dict(P); Q[k] = v
                if f(Q) < best - 1e-4:
                    best, P = f(Q), Q
            log.write(f"# ronda {ronda} tras {k}: {best:.4f} {json.dumps(P)}\n"); log.flush()
    with open(os.path.join(AQUI, "calib8_sondes.json" if MODE == "sondes" else "calib8.json"), "w") as fh:
        json.dump({"params": P, "distancia": best}, fh, indent=1)
