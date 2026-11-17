"""Probabilidades simétricas de una TCN (sus pliegues) sobre las features de otro banco, para evaluar el modelo completo
en un banco en el que no se entrenó. Cada partido de IDSSE con el pliegue que lo dejó fuera; Metrica con el pliegue metrica.
    MTAH_DEG=<banco origen> python tcn_inf_otro.py <dir tcn_models> <dir salida> [desplazamiento de semilla] [fuentes]
    (venv bench, con torch). El desplazamiento renumera las semillas (p. ej. 3: s0-2 -> s3-5), como en el banco v12."""
import os, sys, glob, re
import numpy as np
AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ULISES_BASE") or (os.path.join(AQUI, "base") if os.path.isdir(os.path.join(AQUI, "base")) else os.path.join(AQUI, "..", "base"))
sys.path.insert(0, os.path.join(BASE, "bench")); sys.path.insert(0, BASE)
import train_tcn as tt
from build_labels import desa_npz
from models import tcn_v7
M, OUT = sys.argv[1], sys.argv[2]
DESPL = int(sys.argv[3]) if len(sys.argv) > 3 else 0
FONTS = tuple(sys.argv[4].split(",")) if len(sys.argv) > 4 else ("idsse", "metrica")
for font in FONTS:
    for f in sorted(glob.glob(os.path.join(tt.ft.SORTIDA_EXT, font, "*.npz"))):
        p, ll = re.match(r"(.*)_s(\d+)\.npz$", os.path.basename(f)).groups()
        ck = os.path.join(M, f"plec_idsse_{p}.pt" if font == "idsse" else "plec_metrica.pt")
        ps = tt.probs_extern_sim(ck, font, p, int(ll))
        desa_npz(os.path.join(OUT, font, f"{p}_s{int(ll) + DESPL}.npz"), {"probs": ps, "classes": np.array(tcn_v7.CLASSES)})
    print(font, flush=True)
