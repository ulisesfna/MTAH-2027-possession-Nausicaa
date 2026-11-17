"""Rutas y carga común (partido y externos) reutilizando bench/adapter.py de Jan (rama v9-markius exportada)."""
import os, sys, glob
import numpy as np
AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(AQUI, "..", "base")
ROOT = os.path.abspath(os.path.join(AQUI, "..", ".."))
os.environ.setdefault("MTAH_DEG", "v3")
os.environ.setdefault("MTAH_DATA_DIR", os.path.join(ROOT, "data"))
for p in (BASE, os.path.join(BASE, "bench"), os.path.join(BASE, "evaluation")):
    sys.path.insert(0, os.path.abspath(p))
import adapter  # noqa: E402
SH = os.environ.get("MTAH_SHARED", "/workspace/shared")   # carpeta común de datos del equipo
NET = f"{SH}/bench_v7"            # externos unificados sin degradar (Jan)
V3 = f"{SH}/bench_v7d_deg"        # degradación v3 de Jan
V8 = os.environ.get("MTAH_V8") or os.path.expanduser(os.environ.get("MTAH_LOCAL", "~/mtah_data") + "/bench_v8")   # degradación propia (NO compartida)
_partit = {}

def partit():
    if "d" not in _partit:
        _partit["d"] = adapter.carrega_partit()
    return _partit["d"]

def externs(arrel, font, llavor=0):
    return sorted(glob.glob(f"{arrel}/{font}/*_s{llavor}.npz"))
