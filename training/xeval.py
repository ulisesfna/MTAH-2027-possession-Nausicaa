"""E20: predice con los pliegues de un stack entrenado en un banco (p. ej. v11) sobre las entradas de otro banco (p. ej.
v10 semillas 0-2) y deja las probabilidades donde exp_stack.py las busca, para evaluarlo allí con el mismo protocolo.
    MTAH_DEG=v10 ULISES_PROBS_SUF=_sim ULISES_LLAVORS=0,1,2 python xeval.py <dir del stack origen> <nombre destino> <grupos>"""
import os, sys
import numpy as np, lightgbm as lgb
import lab
from lab import *
import feats_stack as fs
src, nom, g = sys.argv[1], sys.argv[2], tuple(sys.argv[3].split(","))
out = os.path.join(lab.CACHE, f"stack_{nom}")
for font, p in SELECCIO:
    m = lgb.Booster(model_file=os.path.join(src, f"model_{p}.txt" if font == "idsse" else "model_tots.txt"))
    for ll in lab.LLAVORS:
        e = entrada(font, p, ll); feats(e)
        a = m.predict(fs.build(e, False, g)[0]); b = m.predict(fs.build(e, True, g)[0])[:, [1, 0, 2]]
        os.makedirs(f"{out}/{font}", exist_ok=True); np.save(f"{out}/{font}/{p}_s{ll}.npy", ((a + b) / 2).astype(np.float32)); e.pop("X", None)
    print(font, p, flush=True)
