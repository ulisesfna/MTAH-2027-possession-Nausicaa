"""Rutes de cada versió del banc.

MTAH_DEG tria la degradació: v1 (C1 a C3), v2 (C2b, forats de pilota d'IDSSE condicionats a la
línia i als jugadors) o v3 (C2c, a més segons si la pilota és viva o morta). Per defecte, v2. A la
v2 i la v3 només canvia IDSSE; SkillCorner i Metrica són enllaços als fitxers de la v1.
"""
import os

DEG = os.environ.get("MTAH_DEG", "v2")
# Bancos propios (Ulises: v8, v9, v10, v10j...): cualquier MTAH_DEG que no sea v1-v3 vive entero en ~/mtah_data, nada en /workspace/shared
PROPI = DEG not in ("v1", "v2", "v3")
LOCAL = os.path.expanduser(f"~/mtah_data/{DEG}")
DADES = {"v1": "/workspace/shared/bench_v7_deg", "v2": "/workspace/shared/bench_v7b_deg",
         "v3": "/workspace/shared/bench_v7d_deg"}.get(DEG, os.path.expanduser(f"~/mtah_data/bench_{DEG}"))
FEATS = {"v1": "/workspace/shared/features_v7", "v2": "/workspace/shared/features_v7b",
         "v3": "/workspace/shared/features_v7d"}.get(DEG, LOCAL + "/features")
