"""Copia corregida de un directorio de npz de SkillCorner del banco del equipo: invierte players_team (que está al
revés que las etiquetas H/A; bench/build_labels.py línea 461). No modifica el origen.
    python fix_equips_dir.py <dir origen> <dir destino>"""
import glob, json, os, sys
import numpy as np
from common import *
from build_labels import desa_npz
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in sorted(glob.glob(os.path.join(src, "*.npz"))):
    z = np.load(f, allow_pickle=False); d = {k: z[k] for k in z.files}
    d["players_team"] = (1 - d["players_team"]).astype(np.int8)
    m = json.loads(str(d["meta"])); m["equips_corregits"] = True; d["meta"] = np.array(json.dumps(m, sort_keys=True))
    desa_npz(os.path.join(dst, os.path.basename(f)), d)
print(len(glob.glob(os.path.join(dst, "*.npz"))), "ficheros en", dst)
