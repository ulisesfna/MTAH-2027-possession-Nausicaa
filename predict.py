"""Nausicaa: ATD tracking of BAR-ATH -> per-frame possession CSV (frame,label). Single inference script.

    python predict.py --data /path/to/data --out submission.csv

--data is the folder of the competition data bundle that contains tactical/BAR-ATH_tactical_atd.{txt,xml}
(default: $MTAH_DATA_DIR, else ./data). No label is read. CPU, one thread, fixed seeds.
Steps: tracking -> 35 features -> TCN (as is + teams swapped) -> rules -> LightGBM stack -> Viterbi -> CSV.
The model folder (default models/nausicaa) holds tcn.pt, stack.txt and params.json (decoder settings, HMM
statistics, feature groups of the stack).
"""
import argparse, hashlib, json, os, sys

AQUI = os.path.dirname(os.path.abspath(__file__))
# Shared code used at inference (bench/, models/, starter/, games.py) lives in ./base; ULISES_BASE overrides it.
_DENTRO = os.path.join(AQUI, "base")
BASE = os.environ.get("ULISES_BASE") or (_DENTRO if os.path.isdir(_DENTRO) else os.path.dirname(AQUI))


def _raiz(d):
    """Folder whose data/ is the default data location: MTAH_ROOT, else the first parent with CHALLENGE.md, else here."""
    if os.environ.get("MTAH_ROOT"):
        return os.environ["MTAH_ROOT"]
    while True:
        if os.path.exists(os.path.join(d, "CHALLENGE.md")):
            return d
        if os.path.dirname(d) == d:
            return AQUI
        d = os.path.dirname(d)


REPO = _raiz(AQUI)
os.environ.setdefault("MTAH_DEG", "v3")          # solo fija rutas del banco; la inferencia no las usa
os.environ.setdefault("MTAH_DATA_DIR", os.path.join(REPO, "data"))
for p in (AQUI, BASE, os.path.join(BASE, "bench"), os.path.join(BASE, "evaluation")):
    sys.path.insert(0, p)

import numpy as np
import torch
import lightgbm as lgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join("models", "nausicaa"))
    ap.add_argument("--data", default=None, help="data bundle folder containing tactical/ (default: $MTAH_DATA_DIR or ./data)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--comprueba", action="store_true")
    ap.add_argument("--guarda", help="npz con las señales intermedias (diagnóstico sin etiquetas; no se sube)")
    a = ap.parse_args()
    if a.data:
        os.environ["MTAH_DATA_DIR"] = os.path.abspath(a.data)
    mdir = a.model if os.path.isabs(a.model) else os.path.join(AQUI, a.model)
    out = os.path.abspath(a.out or os.path.join(AQUI, "out", f"{os.path.basename(mdir)}.csv"))
    with open(os.path.join(mdir, "params.json")) as fh:
        par = json.load(fh)
    cfg, st, grupos = par["cfg"], par["st"], tuple(par["grupos"])

    torch.manual_seed(0); np.random.seed(0)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    cwd = os.getcwd(); os.chdir(BASE)             # games.py resuelve rutas relativas a la raíz del código base
    import adapter, features as ft, tune_decoder as td
    from models import tcn_v7, decoder as dc
    import feats_stack as fs

    d = adapter.carrega_partit()
    if par.get("reglas", "").endswith("p2_notag"):   # rules without the "mid-flight" retiming of turnovers (retime=None)
        from starter import principled_v1 as pv
        cfg_r = {**pv.CONFIGS["p3_notag"], "retime": None, "exposa_control": True}
        kin = pv.kinematics_from_arrays(d["px"].copy(), d["py"].copy(), d["mean_speed"].copy(), d["bx"].copy(), d["by"].copy(), d["team_of_track"], d["pitch"])
        lab_r, info_r = pv.predict(kin, {"home": None, "away": None, "set_pieces": None}, d["n_frames"], d["half_blocks"], d["kickoff_2_frame"], d["pitch"], cfg_r)
        d["_kin"] = kin; d["_janv1"] = (lab_r, info_r["control_clar"])
    labels, clar = td.janv1_amb_control(d)
    conf = td.conf_fisica(d)
    X, noms, _ = ft.features(d)
    os.chdir(cwd)
    model, ck = tcn_v7.carrega(os.path.join(mdir, "tcn.pt"))
    p_tcn = tcn_v7.probabilitats(model, ck, X)
    if par.get("tcn_simetrica"):     # v10/E17: media con la predicción de los equipos intercambiados (misma permutación que el entrenamiento)
        nn = [str(n) for n in noms]
        idx = [nn.index("vis_" + n[4:]) if n.startswith("loc_") else nn.index("loc_" + n[4:]) if n.startswith("vis_") else i for i, n in enumerate(nn)]
        gira = {"pilota_x", "pilota_y", "pilota_ultima_x", "pilota_ultima_y", "loc_centre_x", "loc_centre_y", "vis_centre_x", "vis_centre_y"}
        sg = np.array([-1.0 if n in gira else 1.0 for n in nn], np.float32)
        p_sw = tcn_v7.probabilitats(model, ck, np.ascontiguousarray(X[:, idx] * sg))[:, [1, 0, 2]]
        p_tcn = ((p_tcn + p_sw) / 2).astype(np.float32)
    period = np.asarray(d["period"])

    e = {"X": X.astype(np.float32), "noms": [str(n) for n in noms], "p": np.round(p_tcn.astype(np.float64), 5),
         "janv1": np.asarray(labels).astype("<U1"), "clar": np.asarray(clar), "phys_conf": conf.astype(np.float32)}
    m = lgb.Booster(model_file=os.path.join(mdir, "stack.txt"))
    Fa, names = fs.build(e, False, grupos); Fb, _ = fs.build(e, True, grupos)
    assert names == m.feature_name(), "las features no coinciden con las del modelo"
    p = np.round((m.predict(Fa, num_threads=1) + m.predict(Fb, num_threads=1)[:, [1, 0, 2]]) / 2, 5)

    prior = np.array(st["prior"])
    em = cfg["a"] * (np.log(np.maximum(p, 1e-6)) - np.log(prior)[None])
    em[:, 2] += cfg.get("d", 0.0) * (e["janv1"] == "D")
    mm, lq, ls = dc.parametres_hmm(st, cfg["s"])
    y = np.full(len(p), 3, np.int8)
    for h in (1, 2):
        idx = np.flatnonzero(period == h); i0, i1 = idx[0], idx[-1] + 1
        y[i0:i1] = dc.viterbi(em[None, i0:i1], mm[None], lq[None], ls[None], np.log(prior))[0]

    if a.guarda:
        np.savez_compressed(a.guarda, p_tcn=p_tcn, p_stack=p, y=y, janv1=e["janv1"], clar=e["clar"], period=period,
                            r=np.isfinite(d["bx"]) & np.isfinite(d["by"]), phys_conf=e["phys_conf"], X=e["X"], noms=np.array(e["noms"]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.write("\n".join(f"{i},{'HADX'[c]}" for i, c in enumerate(y)) + "\n")
    sha = hashlib.sha256(open(out, "rb").read()).hexdigest()
    with open(out + ".sha256", "w") as fh:
        fh.write(f"{sha}  {os.path.basename(out)}\n")
    en = period > 0
    res = {"csv": out, "sha256": sha, "frames": int(len(y)), "pct": {c: round(100 * float(np.mean(y[en] == i)), 2) for i, c in enumerate("HAD")},
           "sha_p_tcn_5dec": hashlib.sha256(np.round(p_tcn, 5).tobytes()).hexdigest()[:16], "sha_p_stack_5dec": hashlib.sha256(p.tobytes()).hexdigest()[:16]}
    if a.comprueba:
        deg = par.get("banco", "v3")
        cache = os.path.join(AQUI, f"cache_{deg}" if deg != "v3" else "cache", "partit")
        if os.path.exists(os.path.join(cache, "entrada.npz")):
            z = np.load(os.path.join(cache, "entrada.npz"))
            res["tcn_max_dif_vs_lab"] = float(np.abs(z["p"] - p_tcn).max()); res["reglas_iguales_lab"] = bool((z["janv1"].astype("<U1") == e["janv1"]).all())
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
