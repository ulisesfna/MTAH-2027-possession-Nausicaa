"""Validación por comportamiento de modelos, SIN etiquetas del partido.

Pasa los mismos modelos fijos por el BAR-ATH y por los partidos de IDSSE de un banco (v3 de Jan o v8) y mide cosas
que no necesitan GT: % de balón muerto que predice cada uno, posesiones por hora, D con balón visible y acuerdo
entre modelos. Si el banco se parece al partido, cada modelo debe comportarse igual en los dos.

Modelos sonda: reglas janv1 p3_notag, baseline_smooth50, TCN v7d (argmax; en IDSSE el modelo del pliegue que no ha
visto el partido, en el BAR-ATH el final) y fusio_v7d (TCN + reglas + Viterbi con los parámetros del pliegue 'tots').

    /workspace/shared/venvs/bench/bin/python probes.py partit|v3|v8  -> probes_<x>.json
"""
import json, sys
import numpy as np
from common import *
import torch
import features as ft, tune_decoder as td, harness as hz
from models import tcn_v7, decoder as dc
torch.set_num_threads(2)
C = {"H": 0, "A": 1, "D": 2, "X": 3}
cod = lambda lab: np.vectorize(C.get)(np.asarray(lab).astype("<U1"))


def spells_h(y, per):
    n = 0
    for p in (1, 2):
        w = y[per == p]
        if len(w):
            ch = np.flatnonzero(w[1:] != w[:-1]) + 1
            st = np.concatenate(([0], ch)); n += int(np.sum(w[st] < 2))
    return n / ((per > 0).sum() / 25 / 3600)


def sondes(d, pesos, gt=None):
    per = np.asarray(d["period"]); m = per > 0
    vis = np.isfinite(d["bx"]) & m
    jv, clar = td.janv1_amb_control(d); jv = cod(jv)
    bs = cod(hz.pred_baseline_smooth50(d))
    model, ck = tcn_v7.carrega(pesos); p = tcn_v7.probabilitats(model, ck, ft.features(d)[0]); tc = p.argmax(1)
    par = td.ajusta("fusio", set(td.plecs()["tots"][1]), "v7d", "c2c")
    fu = cod(dc.descodifica(p, np.vectorize("HADX".__getitem__)(jv), ~clar, per, par["st"], [par["cfg"]])[0])
    out = {"visible": float(vis.sum() / m.sum())}
    for nom, y in (("regles", jv), ("baseline", bs), ("tcn", tc), ("fusio", fu)):
        out[f"pctD_{nom}"] = float(np.mean(y[m] == 2)); out[f"spells_h_{nom}"] = spells_h(y, per)
        out[f"D_vis_{nom}"] = float(np.sum((y == 2) & vis) / m.sum()); out[f"viu_novis_{nom}"] = float(np.sum((y < 2) & m & ~vis) / m.sum())
        out[f"pH_{nom}"] = float(np.mean(y[m] == 0) / max(np.mean(y[m] < 2), 1e-9))
    for a, ya, b, yb in (("regles", jv, "tcn", tc), ("regles", jv, "fusio", fu), ("regles", jv, "baseline", bs), ("tcn", tc, "fusio", fu)):
        out[f"acord_{a}_{b}"] = float(np.mean(ya[m] == yb[m])); out[f"acord_vis_{a}_{b}"] = float(np.mean(ya[vis] == yb[vis]))
    out["difD_tcn_regles"] = out["pctD_tcn"] - out["pctD_regles"]; out["difDvis_tcn_regles"] = out["D_vis_tcn"] - out["D_vis_regles"]
    out["conf_tcn"] = float(p[m].max(1).mean()); out["conf_tcn_vis"] = float(p[vis].max(1).mean())
    if gt is not None:   # solo en externos: nota real de cada sonda
        import possession_metrics as pm
        g = hz.gt_avaluable(gt, "label_acc")[0]
        for nom, y in (("regles", jv), ("baseline", bs), ("tcn", tc), ("fusio", fu)):
            r = pm.evaluate(g, np.where(m, y, 3)); out[f"spellF1_{nom}"] = r["spell_f1_mean"]; out[f"macroF1_{nom}"] = r["macro_f1"]
        out["pctD_gt"] = float(np.mean(g[g != 3] == 2))
    return out


if __name__ == "__main__":
    que = sys.argv[1]
    res = {}
    if que == "partit":
        res["BAR-ATH"] = sondes(partit(), os.path.join(BASE, "models", "tcn_v7d_weights.pt"))
    else:
        arrel = {"v3": V3, "v8": V8}.get(que, que)
        for f in externs(arrel, "idsse"):
            d = adapter.carrega_extern(f)
            res[d["partit"]] = sondes(d, f"{SH}/tcn_v7d_models/plec_idsse_{d['partit']}.pt", d["gt"]); print(d["partit"], flush=True)
    with open(os.path.join(AQUI, f"probes_{os.path.basename(que)}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
