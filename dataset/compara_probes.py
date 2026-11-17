"""Tabla: comportamiento de cada sonda en el BAR-ATH frente a IDSSE v3 y v8 (media ± desv. entre los 7 partidos) y
z = (partido − media banco) / desv. Cuanto más cerca de 0, más se comporta el modelo en el banco como en el partido."""
import json, numpy as np, sys
m = json.load(open("probes_partit.json"))["BAR-ATH"]
B = {v: json.load(open(f"probes_{v}.json")) for v in sys.argv[1:] or ("v3", "v8")}
keys = [k for k in m if not k.startswith(("spellF1", "macroF1", "pctD_gt"))]
print(f"{'':26s}{'BAR-ATH':>9s}" + "".join(f"{v + ' media±sd':>18s}{'z':>7s}" for v in B))
zs = {v: [] for v in B}
for k in keys:
    s = f"{k:26s}{m[k]:9.3f}"
    for v, d in B.items():
        x = np.array([r[k] for r in d.values()]); z = (m[k] - x.mean()) / max(x.std(ddof=1), 1e-9)
        zs[v].append(abs(z)); s += f"{x.mean():11.3f}±{x.std(ddof=1):5.3f}{z:7.1f}"
    print(s)
print("\n|z| medio / mediano / nº de |z|>2:", {v: (round(float(np.mean(z)), 2), round(float(np.median(z)), 2), int(np.sum(np.array(z) > 2))) for v, z in zs.items()})
print("\nNotas reales en el banco (label_acc, 7 IDSSE, semilla 0):")
for k in ("spellF1_regles", "spellF1_baseline", "spellF1_tcn", "spellF1_fusio", "macroF1_regles", "macroF1_tcn", "macroF1_fusio", "pctD_gt"):
    print(f"  {k:18s}" + "".join(f"  {v}: {np.mean([r[k] for r in d.values()]):.3f}" for v, d in B.items()))
