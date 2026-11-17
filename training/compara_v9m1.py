"""Compara el modelo entregado con v9m1 y fusio_v7d del equipo en el banco de referencia del equipo, por partido y con
bootstrap pareado (cifras del documento técnico, sección 2 y figura 2). Escribe también los datos de la figura 2.
    MTAH_DEG=v3 python compara_v9m1.py <results_v3view_sim/stack_c11_g3.json> <informe JSON del arnés del equipo> [fig_data.json]
El informe del arnés es bench/results/2026-11-16_113518_markius_v9_m1x-markius_v9_m1-markius_v9-fusio_v7d_s012_sensesuau.json
de la rama v9m1-regles del equipo."""
import json, sys
import numpy as np
from lab import bootstrap
u = json.load(open(sys.argv[1]))["partits"]; jan = json.load(open(sys.argv[2]))["principal"]
ks = sorted(u)
vec = lambda name: np.array([jan[name]["per_partit"][k]["spellF1"] for k in ks])
vu = np.array([u[k]["acc"] for k in ks]); vm1 = vec("markius_v9_m1"); vf = vec("fusio_v7d")
print(f"spell-F1 label_acc, 10 partidos: entregado {vu.mean():.4f} | v9m1 {vm1.mean():.4f} | fusio_v7d {vf.mean():.4f}")
for nom, d in (("entregado - v9m1", vu - vm1), ("entregado - fusio_v7d", vu - vf)):
    m, lo, hi = bootstrap(list(d)); print(f"{nom}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}], mejor en {int((d > 0).sum())}/10")
for k, a, b in zip(ks, vm1, vu):
    print(f"  {k:28s} v9m1 {a:.3f}  entregado {b:.3f}")
if len(sys.argv) > 3:
    json.dump({"partits": ks, "fusio": vf.tolist(), "v9m1": vm1.tolist(), "ulises": vu.tolist()}, open(sys.argv[3], "w"))
