"""Validación adversarial del banco (sin etiquetas): ¿puede un clasificador distinguir IDSSE degradado del BAR-ATH
usando las 35 features de bench/features.py (lo que ve la TCN) con contexto temporal (t, t±0.5 s, t±1 s)?
Entrena con 4 IDSSE + 1ª parte del partido y evalúa en 3 IDSSE + 2ª parte (no puede memorizar el partido).
    .venv/bin/python adv_ds.py v3 v8   -> AUC (0.5 = indistinguible) y features que más separan."""
import sys, json
import numpy as np, lightgbm as lgb
from common import *
import types; sys.modules.setdefault("yaml", types.ModuleType("yaml"))  # .venv no tiene yaml y features.py no lo usa
import features as ft
rng = np.random.RandomState(0)
FORA = {"part_2", "en_part", "temps_part", "pilota_vista_part", "loc_te_jugadors", "loc_te_2", "vis_te_jugadors", "vis_te_2"}


def mostra(d, per_sel, n):
    X, noms = ft.features(d)[0], None
    z = np.load(externs(V3, "idsse")[0].replace(V3, f"{SH}/features_v7d"))
    noms = [str(x) for x in z["noms"]]
    keep = [i for i, c in enumerate(noms) if c not in FORA]
    X = X[:, keep]; noms = [noms[i] for i in keep]
    cols, names = [X], list(noms)
    for k in (-25, -12, 12, 25):
        cols.append(np.roll(X, -k, axis=0)); names += [f"{c}{k:+d}" for c in noms]
    F = np.concatenate(cols, 1)
    idx = np.flatnonzero(np.isin(np.asarray(d["period"]), per_sel)); idx = idx[(idx > 30) & (idx < len(F) - 30)]
    return F[rng.choice(idx, min(n, len(idx)), replace=False)], names


def auc(p, y):
    r = np.argsort(np.argsort(p)).astype(float); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


FONT = os.environ.get("ADV_FONT", "idsse"); SORTIDA = os.environ.get("ADV_OUT", "adv_ds.json")
dm = partit()
Mtr, names = mostra(dm, (1,), 30000); Mte, _ = mostra(dm, (2,), 30000)
res = {}
for v in sys.argv[1:]:
    fs = externs({"v3": V3, "v8": V8}.get(v, v), FONT)
    Btr = np.concatenate([mostra(adapter.carrega_extern(f), (1, 2), 7500)[0] for f in fs[:4]])
    Bte = np.concatenate([mostra(adapter.carrega_extern(f), (1, 2), 10000)[0] for f in fs[4:]])
    Xtr = np.concatenate([Btr, Mtr]); ytr = np.r_[np.zeros(len(Btr)), np.ones(len(Mtr))]
    Xte = np.concatenate([Bte, Mte]); yte = np.r_[np.zeros(len(Bte)), np.ones(len(Mte))]
    m = lgb.train(dict(objective="binary", num_leaves=15, learning_rate=0.1, min_data_in_leaf=300, feature_fraction=0.7, verbose=-1, num_threads=2, seed=0), lgb.Dataset(Xtr, ytr, feature_name=names), 150)
    a = auc(m.predict(Xte), yte)
    imp = sorted(zip(m.feature_importance("gain"), names), reverse=True)[:10]
    # AUC univariante de las features sin desplazar
    uni = sorted(((abs(auc(np.r_[Bte[:, j], Mte[:, j]] + 1e-9 * rng.rand(len(yte)), yte) - 0.5), n) for j, n in enumerate(names[:len(names) // 5])), reverse=True)[:8]
    res[v] = {"auc": a, "top_gain": [n for _, n in imp], "top_univariante": [(n, round(x, 3)) for x, n in uni]}
    print(v, json.dumps(res[v], ensure_ascii=False), flush=True)
json.dump({os.path.basename(k): v for k, v in res.items()}, open(os.path.join(AQUI, SORTIDA), "w"), indent=1)
