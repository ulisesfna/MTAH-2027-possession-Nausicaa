"""Empaqueta el modelo `ulises` entrenado sobre un banco en model_<deg>/ (lo que lee predict.py y lo que se sube al repo).
    MTAH_DEG=v8 .venv/bin/python export_model.py t2 stack_t2_g3 tcn,jv,raw,touch"""
import json, os, shutil, sys
import lab
from lab import *
var, fam, grupos = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
deg = lab.DEG
out = os.path.join(lab.AQUI, f"model_{deg}" + os.environ.get("ULISES_PROBS_SUF", "") + ("_" + var if var != "t2" else "")); os.makedirs(out, exist_ok=True)
pesos = os.path.expanduser(os.environ.get("MTAH_LOCAL", "~/mtah_data") + f"/{deg}/tcn_{deg}_weights.pt") if deg != "v3" else os.path.join(lab.BASE, "models", "tcn_v7d_weights.pt")
shutil.copy(pesos, os.path.join(out, "tcn.pt"))
shutil.copy(f"{lab.CACHE}/stack_{var}/model_tots.txt", os.path.join(out, "stack.txt"))
r = carrega(fam)
json.dump({"banco": deg, "origen": f"results{'_' + deg if deg != 'v3' else ''}/{fam}.json (pliegue tots)", "cfg": r["tria"]["tots"]["cfg"], "st": stats("tots"),
           "grupos": grupos, "tcn_simetrica": os.environ.get("ULISES_PROBS_SUF", "") == "_sim", "tcn": os.path.basename(pesos), "reglas": "janv1 p3_notag"}, open(os.path.join(out, "params.json"), "w"), indent=1)
print(out, os.listdir(out), r["tria"]["tots"]["cfg"])
