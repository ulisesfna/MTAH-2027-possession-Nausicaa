"""E23: reglas p2_notag (= p3_notag con retime=None). Recalcula las etiquetas de las reglas (janv1, clar) de las entradas del decodificador de un banco y las escribe en
otro directorio de entradas (el resto de campos se copia igual). El retiming solo cambia las etiquetas al final de
principled_v1.predict, así que phys_conf, r, period y las etiquetas externas no cambian.
    python e23_reglas_p2.py <dir entrades origen> <dir entrades destino> <dir banco (npz degradados)>   (venv bench)"""
import os, sys, glob, json
from multiprocessing import Pool
import numpy as np
AQUI = os.path.dirname(os.path.abspath(__file__)); BASE = os.environ.get("ULISES_BASE") or os.path.join(AQUI, "base")
for p in (BASE, os.path.join(BASE, "bench"), os.path.join(BASE, "evaluation")):
    sys.path.insert(0, p)
SRC, DST, BENCH = sys.argv[1], sys.argv[2], sys.argv[3]


def regles_p2(d):
    from starter import principled_v1 as pv
    cfg = {**pv.CONFIGS["p3_notag"], "retime": None, "exposa_control": True}
    kin = pv.kinematics_from_arrays(d["px"].copy(), d["py"].copy(), d["mean_speed"].copy(), d["bx"].copy(), d["by"].copy(), d["team_of_track"], d["pitch"])
    labels, info = pv.predict(kin, {"home": None, "away": None, "set_pieces": None}, d["n_frames"], d["half_blocks"], d["kickoff_2_frame"], d["pitch"], cfg)
    return np.asarray(labels).astype("<U1"), np.asarray(info["control_clar"])


def fes(rel):
    import adapter
    from build_labels import desa_npz
    dst = os.path.join(DST, rel)
    if os.path.exists(dst):
        return rel, "ja"
    z = np.load(os.path.join(SRC, rel), allow_pickle=False); e = {k: z[k] for k in z.files}
    d = adapter.carrega_extern(os.path.realpath(os.path.join(BENCH, rel)))
    lab, clar = regles_p2(d)
    assert len(lab) == len(e["janv1"])
    canvi = float(np.mean(lab != e["janv1"].astype("<U1")))
    e["janv1"] = lab; e["clar"] = clar.astype(e["clar"].dtype)
    desa_npz(dst, e)
    return rel, f"{canvi:.4f}"


if __name__ == "__main__":
    fonts = os.environ.get("E23_FONTS", "idsse,metrica").split(",")      # el stack solo usa IDSSE y Metrica
    rels = sorted(os.path.relpath(f, SRC) for font in fonts for f in glob.glob(os.path.join(SRC, font, "*.npz")))
    with Pool(2) as pool:
        for rel, c in pool.imap(fes, rels):
            print(rel, "frames con etiqueta distinta:", c, flush=True)
