"""v8.2: corrige la codificación de equipos de SkillCorner en ~/mtah_data/bench_v8 (players_team invertido respecto a
las etiquetas H/A en los 20 partidos del banco del equipo: bench/build_labels.py). Idempotente (marca deg_meta)."""
import glob, json, sys
import numpy as np
from common import *
from build_labels import desa_npz
for f in sorted(glob.glob(f"{V8}/skillcorner/*.npz")):
    z = np.load(f, allow_pickle=False); d = {k: z[k] for k in z.files}
    info = json.loads(str(d["deg_meta"]))
    if info.get("equips_corregits"):
        continue
    d["players_team"] = (1 - d["players_team"]).astype(np.int8)
    info["equips_corregits"] = True; info["versio"] = "v8.2"
    d["deg_meta"] = np.array(json.dumps(info, sort_keys=True))
    desa_npz(f, d); print(os.path.basename(f), flush=True)
