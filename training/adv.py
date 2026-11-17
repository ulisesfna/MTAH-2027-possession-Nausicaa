"""Validación adversarial (sin etiquetas): ¿qué features del stack distinguen banco externo de BAR-ATH?
AUC por feature (univariante) y AUC de un LightGBM banco-vs-partido. Solo entradas, ninguna etiqueta del partido."""
import numpy as np, lightgbm as lgb, lab, stack
from lab import *
import match_predict as mp
rng = np.random.RandomState(0)
def filas(e, per, n):
    F, names = stack.build(e, False, stack.GRUPS)
    idx = rng.choice(np.flatnonzero(per), n, replace=False)
    return F[idx], names
em = mp.entrada_partit(); Fm, names = filas(em, em["period"] > 0, 40000)
def auc1(a, b):
    x = np.concatenate([a, b]); r = np.argsort(np.argsort(x, kind="stable")).astype(float)
    u = r[:len(a)].sum() - len(a) * (len(a) - 1) / 2
    return u / (len(a) * len(b))
for font, ps in (("idsse", IDSSE), ("metrica", METRICA)):
    Fb = np.concatenate([filas(entrada(font, p, ll), entrada(font, p, ll)["gt_acc"] != 3, 40000 // len(ps) // 2)[0] for p in ps for ll in (0, 1)])
    X = np.concatenate([Fb, Fm]); y = np.r_[np.zeros(len(Fb)), np.ones(len(Fm))]
    tr = rng.rand(len(y)) < 0.7
    m = lgb.train(dict(objective="binary", num_leaves=15, learning_rate=0.1, verbose=-1, num_threads=1, min_data_in_leaf=200), lgb.Dataset(X[tr], y[tr], feature_name=names), 100)
    pr = m.predict(X[~tr]); print(f"## {font} vs BAR-ATH: AUC multivariante {auc1(pr[y[~tr]==1], pr[y[~tr]==0]):.3f}")
    u = sorted(((abs(auc1(Fm[:, j] + 1e-9 * rng.rand(len(Fm)), Fb[:, j] + 1e-9 * rng.rand(len(Fb))) - 0.5), n, float(np.mean(Fb[:, j])), float(np.mean(Fm[:, j]))) for j, n in enumerate(names)), reverse=True)
    print("  más desplazadas (|AUC-0.5|, feature, media banco, media partido):"); [print(f"   {a:.3f} {n:18s} {b:8.3f} {c:8.3f}") for a, n, b, c in u[:18]]
    imp = sorted(zip(m.feature_importance("gain"), names), reverse=True)[:12]; print("  importancia adversarial:", [n for _, n in imp])
