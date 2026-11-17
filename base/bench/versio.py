"""Rutes de cada versió del banc.

MTAH_DEG tria la degradació: v1 (C1 a C3), v2 (C2b, forats de pilota d'IDSSE condicionats a la
línia i als jugadors) o v3 (C2c, a més segons si la pilota és viva o morta). Per defecte, v2. A la
v2 i la v3 només canvia IDSSE; SkillCorner i Metrica són enllaços als fitxers de la v1.
"""
import os

DEG = os.environ.get("MTAH_DEG", "v2")
DADES = {"v1": "/workspace/shared/bench_v7_deg", "v2": "/workspace/shared/bench_v7b_deg",
         "v3": "/workspace/shared/bench_v7d_deg"}[DEG]
FEATS = {"v1": "/workspace/shared/features_v7", "v2": "/workspace/shared/features_v7b",
         "v3": "/workspace/shared/features_v7d"}[DEG]
