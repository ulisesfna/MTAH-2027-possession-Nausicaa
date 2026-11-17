"""Laboratorio de ulises: experimentos de decodificador sobre el banco de Jan (MTAH_DEG=v3).

Reproduce el protocolo anidado de bench/tune_decoder.py + bench/harness.py con las entradas ya
cacheadas en /workspace/shared (solo lectura):
  - 10 partidos de selección (7 IDSSE + 3 Metrica), semillas 0-2, probabilidades TCN v7d fuera de pliegue.
  - Para cada IDSSE que se deja fuera: estadísticas del HMM de los otros 26 y parámetros elegidos con
    los otros 6 IDSSE (label_acc). Para Metrica: estadísticas de los 27, parámetros con los 7 IDSSE.
  - spellF1 medio por partido (sobre semillas), bootstrap entre partidos (2000, semilla 0), pareado.
Etiquetas: label_acc (principal), label_dur (robustez de Jan) y label_prov (canal del proveedor).

Una "familia" es una función decode(e, st, cfg) -> códigos 0/1/2/3 (n,), y una lista de cfgs (dicts).
"""
import ctypes
import glob
import json
import os
import subprocess
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ULISES_BASE") or next(d for d in (os.path.join(AQUI, "base"), os.path.join(AQUI, "..", "base")) if os.path.isdir(d) or d.endswith(os.path.join("..", "base")))   # ULISES_BASE: copia del código base en otra ruta (PC)
os.environ.setdefault("MTAH_DEG", "v3")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "bench"))
sys.path.insert(0, os.path.join(BASE, "evaluation"))
import possession_metrics as pm  # noqa: E402
from models import decoder as dc  # noqa: E402

SH = os.environ.get("MTAH_SHARED", "/workspace/shared")   # carpeta común de datos del equipo
DEG = os.environ["MTAH_DEG"]
if DEG not in ("v1", "v2", "v3"):     # banco propio (dataset_v8): todo en ~/mtah_data, resultados y cachés aparte
    _L = os.path.expanduser(os.environ.get("MTAH_LOCAL", "~/mtah_data") + f"/{DEG}")
    DEC, PROBS, BENCH, FEATS = f"{_L}/decoder", f"{_L}/tcn_probs", os.path.expanduser(os.environ.get("MTAH_LOCAL", "~/mtah_data") + f"/bench_{DEG}"), f"{_L}/features"
    CACHE, RESULTS = os.path.join(AQUI, f"cache_{DEG}"), os.path.join(AQUI, f"results_{DEG}")
else:
    DEC, PROBS, BENCH, FEATS = f"{SH}/decoder_v7d", f"{SH}/tcn_v7d_probs", f"{SH}/bench_v7d_deg", f"{SH}/features_v7d"
    CACHE, RESULTS = os.path.join(AQUI, "cache"), os.path.join(AQUI, "results")
ENT = f"{DEC}/entrades"
# v10: sufijos opcionales. ULISES_PROBS_SUF="_sim" lee las probabilidades simétricas de la TCN; ULISES_SUF separa cachés y
# resultados de una variante (p. ej. "_sim", "_s345"); ULISES_LLAVORS elige las semillas que se evalúan.
CACHE_BASE = CACHE          # las etiquetas cacheadas (gt_prov) no dependen de la variante
PROBS += os.environ.get("ULISES_PROBS_SUF", "")
CACHE += os.environ.get("ULISES_CACHE_SUF", os.environ.get("ULISES_PROBS_SUF", ""))
RESULTS += os.environ.get("ULISES_SUF", os.environ.get("ULISES_PROBS_SUF", ""))
LLAVORS = tuple(int(x) for x in os.environ.get("ULISES_LLAVORS", "0,1,2").split(","))
X = 3

# ------------------------------------------------------------------ Viterbi en C
_so = os.path.join(AQUI, "cvit.so")
_lib = None
try:   # sin gcc (p. ej. Windows) el laboratorio se importa igual; solo falla quien llame a viterbi()
    if not os.path.exists(_so) or os.path.getmtime(_so) < os.path.getmtime(os.path.join(AQUI, "cvit.c")):
        subprocess.check_call(["gcc", "-O2", "-shared", "-fPIC", "-o", _so, os.path.join(AQUI, "cvit.c"), "-lm"])
    _lib = ctypes.CDLL(_so)
    _dp = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    _lib.viterbi.argtypes = [_dp, ctypes.c_long, np.ctypeslib.ndpointer(np.int64, flags="C_CONTIGUOUS"), _dp, _dp, _dp,
                             ctypes.c_void_p, np.ctypeslib.ndpointer(np.int8, flags="C_CONTIGUOUS")]
except Exception as ex:   # noqa: BLE001
    print("aviso: Viterbi en C no disponible:", ex, file=sys.stderr)


def viterbi(e, m, lq, ls, li, xpen=None):
    e = np.ascontiguousarray(e, np.float64)
    out = np.empty(len(e), np.int8)
    xp = None
    if xpen is not None:
        xpen = np.ascontiguousarray(xpen, np.float64)
        xp = xpen.ctypes.data_as(ctypes.c_void_p)
    _lib.viterbi(e, len(e), np.ascontiguousarray(m, np.int64), np.ascontiguousarray(lq, np.float64),
                 np.ascontiguousarray(ls, np.float64), np.ascontiguousarray(li, np.float64), xp, out)
    return out


# ------------------------------------------------------------------ datos
def partits(font):
    return sorted({os.path.basename(f).rsplit("_s", 1)[0] for f in glob.glob(f"{ENT}/{font}/*_s0.npz")})


IDSSE = partits("idsse")
METRICA = partits("metrica")
SELECCIO = [("idsse", p) for p in IDSSE] + [("metrica", p) for p in METRICA]
_mem = {}


def _gt_prov(font, partit, ll):
    """label_prov evaluable (misma máscara que harness.gt_avaluable), cacheado en local."""
    path = os.path.join(CACHE_BASE, "gt_prov", font, f"{partit}_s{ll}.npy")
    if not os.path.exists(path):
        import harness as hz
        z = np.load(f"{BENCH}/{font}/{partit}_s{ll}.npz", allow_pickle=False)
        g = hz.gt_avaluable({k: z[k] for k in ("label_prov", "valid", "period")}, "label_prov")[0]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, g.astype(np.int8))
    return np.load(path)


def entrada(font, partit, ll):
    k = (font, partit, ll)
    if k not in _mem:
        z = np.load(f"{ENT}/{font}/{partit}_s{ll}.npz", allow_pickle=False)
        e = {x: z[x] for x in z.files}
        e["p"] = np.load(f"{PROBS}/{font}/{partit}_s{ll}.npz", allow_pickle=False)["probs"].astype(np.float64)
        e["gt_prov"] = _gt_prov(font, partit, ll)
        e["janv1"] = e["janv1"].astype("<U1")
        e["q"] = dc.q_janv1(e["janv1"])
        e["key"] = k
        _mem[k] = e
    return _mem[k]


def feats(e):
    """Matriz de 35 features de bench/features.py (carga perezosa)."""
    if "X" not in e:
        font, partit, ll = e["key"]
        z = np.load(f"{FEATS}/{font}/{partit}_s{ll}.npz", allow_pickle=False)
        e["X"] = z["X"]
        e["noms"] = [str(n) for n in z["noms"]]
    return e["X"]


def stats(nom):
    with open(f"{DEC}/estadistiques_{nom}.json") as fh:
        return json.load(fh)


def plec_de(font, partit):
    return f"idsse_{partit}" if font == "idsse" else "tots"


def ajust_de(nom):
    return [("idsse", q) for q in IDSSE if nom != f"idsse_{q}"]


# ------------------------------------------------------------------ métrica rápida (= pm.spell_scores)
def _spells(codes, runs):
    out = []
    for a, b in runs:
        w = codes[a:b]
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        bo = np.concatenate(([0], br, [len(w)]))
        for i in range(len(bo) - 1):
            k = int(w[bo[i]])
            if k in (0, 1):
                out.append((a + int(bo[i]), a + int(bo[i + 1]), k))
    return out


def spell_f1(gt, pred, detall=False):
    runs = pm.valid_runs(gt)
    pred = pm.fill_abstentions(pred, runs)
    g, p = _spells(gt, runs), _spells(pred, runs)
    ov = []
    j0 = 0
    for i, (ga, gb, gk) in enumerate(g):
        while j0 < len(p) and p[j0][1] <= ga:
            j0 += 1
        j = j0
        while j < len(p) and p[j][0] < gb:
            pa, pb, pk = p[j]
            if pk == gk:
                ov.append(((min(gb, pb) - max(ga, pa)) / float(max(gb, pb) - min(ga, pa)), i, j))
            j += 1
    ov.sort(key=lambda t: (-t[0], t[1], t[2]))
    f1s = []
    for thr in pm.SPELL_IOUS:
        fg, fp = set(), set()
        for iou, i, j in ov:
            if iou < thr:
                break
            if i in fg or j in fp:
                continue
            fg.add(i)
            fp.add(j)
        tp = len(fg)
        pr = tp / len(p) if p else 0.0
        rc = tp / len(g) if g else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    if detall:
        return float(np.mean(f1s)), len(g), len(p)
    return float(np.mean(f1s))


def macro_f1(gt, pred):
    ok = gt != X
    f = []
    for k in range(3):
        tp = np.sum((gt == k) & (pred == k) & ok)
        fp = np.sum((gt != k) & (pred == k) & ok)
        fn = np.sum((gt == k) & (pred != k) & ok)
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f))


# ------------------------------------------------------------------ decodificador base (fusio de Jan)
def hmm(st, s_dur, s_dur_d=None):
    m, lq, ls = dc.parametres_hmm(st, s_dur)
    if s_dur_d is not None:
        m2, lq2, ls2 = dc.parametres_hmm(st, s_dur_d)
        m, lq, ls = m.copy(), lq.copy(), ls.copy()
        m[2], lq[2], ls[2] = m2[2], lq2[2], ls2[2]
    return m, lq, ls


def per_parts(e, fn):
    """Aplica fn(a, b) -> códigos a cada parte; fuera, X."""
    out = np.full(len(e["period"]), X, np.int8)
    for h in (1, 2):
        idx = np.flatnonzero(e["period"] == h)
        if idx.size:
            a, b = idx[0], idx[-1] + 1
            out[a:b] = fn(a, b)
    return out


def emis_fusio(e, st, cfg):
    prior = np.array(st["prior"])
    w = cfg["w"]
    if cfg.get("porta"):
        w = np.where(~e["clar"], w, 0.0)
    return dc.emissions(e["p"], e["q"], prior, w, cfg["lam"])


def decode_fusio(e, st, cfg):
    em = emis_fusio(e, st, cfg)
    m, lq, ls = hmm(st, cfg["s_dur"])
    li = np.log(np.array(st["prior"]))
    return per_parts(e, lambda a, b: viterbi(em[a:b], m, lq, ls, li))


GRAELLA_FUSIO = [{"w": w, "lam": l, "s_dur": s} for w in (0.5, 0.75, 1.0) for l in (0.05, 0.1, 0.15, 0.25) for s in (0.5, 1.0)]


# ------------------------------------------------------------------ protocolo anidado
def bootstrap(v, n=2000, llavor=0):
    v = np.asarray(v, float)
    rng = np.random.RandomState(llavor)
    ms = v[rng.randint(0, len(v), (n, len(v)))].mean(axis=1)
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def avalua_familia(nom, decode, cfgs, verbose=True, desa=True):
    """Selección anidada + evaluación. Devuelve dict con spellF1 por partido y etiqueta."""
    # 1) tabla de ajuste: para cada pliegue, score label_acc de cada cfg en sus partidos de ajuste
    taula = {}   # (plec, partit, ll) -> [score por cfg]
    preds = {}   # (plec, partit, ll, icfg) solo si el partido se evalúa con ese pliegue: no hace falta guardar
    plecs = [f"idsse_{p}" for p in IDSSE] + ["tots"]
    tria = {}
    for pl in plecs:
        st = stats(pl)
        sc = []
        for font, q in ajust_de(pl):
            sc.append([[spell_f1(entrada(font, q, ll)["gt_acc"], decode(entrada(font, q, ll), st, c)) for c in cfgs]
                       for ll in LLAVORS])
        mitj = np.mean(np.array(sc), axis=(0, 1))  # media sobre partidos de la media sobre semillas
        best, val = 0, -1.0
        for i, x in enumerate(mitj):
            if x > val + 1e-12:
                best, val = i, float(x)
        tria[pl] = (best, val)
    # 2) evaluación fuera de pliegue
    res = {"nom": nom, "tria": {pl: {"cfg": cfgs[i], "ajust": v} for pl, (i, v) in tria.items()}, "partits": {}}
    for font, p in SELECCIO:
        pl = plec_de(font, p)
        st, cfg = stats(pl), cfgs[tria[pl][0]]
        r = {"acc": [], "dur": [], "prov": [], "macro": [], "n_pred": [], "n_gt": []}
        for ll in LLAVORS:
            e = entrada(font, p, ll)
            y = decode(e, st, cfg)
            s, ng, npd = spell_f1(e["gt_acc"], y, detall=True)
            r["acc"].append(s)
            r["dur"].append(spell_f1(e["gt_dur"], y))
            r["prov"].append(spell_f1(e["gt_prov"], y))
            r["macro"].append(macro_f1(e["gt_acc"], y))
            r["n_pred"].append(npd)
            r["n_gt"].append(ng)
        res["partits"][f"{font}/{p}"] = {k: float(np.mean(v)) for k, v in r.items()}
    for k in ("acc", "dur", "prov", "macro"):
        res[k] = bootstrap([v[k] for v in res["partits"].values()])
    if desa:
        os.makedirs(os.path.join(RESULTS), exist_ok=True)
        with open(os.path.join(RESULTS, nom + ".json"), "w") as fh:
            json.dump(res, fh, indent=1)
    if verbose:
        mostra(res)
    return res


def carrega(nom):
    with open(os.path.join(RESULTS, nom + ".json")) as fh:
        return json.load(fh)


def compara(a, b, k="acc"):
    """b - a pareado por partido: media, IC95, nº de partidos donde b > a."""
    ks = sorted(a["partits"])
    d = [b["partits"][x][k] - a["partits"][x][k] for x in ks]
    m, lo, hi = bootstrap(d)
    return m, lo, hi, int(sum(x > 0 for x in d))


def mostra(res, base=None):
    print(f"== {res['nom']}: acc {res['acc'][0]:.4f} [{res['acc'][1]:.3f},{res['acc'][2]:.3f}]  dur {res['dur'][0]:.4f}  "
          f"prov {res['prov'][0]:.4f}  macro {res['macro'][0]:.4f}")
    ids = [v for k, v in res["partits"].items() if k.startswith("idsse")]
    met = [v for k, v in res["partits"].items() if k.startswith("metrica")]
    print("   IDSSE acc/dur/prov {:.4f}/{:.4f}/{:.4f}   Metrica acc/dur/prov {:.4f}/{:.4f}/{:.4f}   n_pred/n_gt {:.0f}/{:.0f}".format(
        *[np.mean([v[k] for v in ids]) for k in ("acc", "dur", "prov")],
        *[np.mean([v[k] for v in met]) for k in ("acc", "dur", "prov")],
        np.mean([v["n_pred"] for v in res["partits"].values()]), np.mean([v["n_gt"] for v in res["partits"].values()])))
    cf = {}
    for pl, t in res["tria"].items():
        cf.setdefault(json.dumps(t["cfg"], sort_keys=True), []).append(pl.replace("idsse_DFL-MAT-", ""))
    for c, pls in cf.items():
        print("   cfg", c, "x", len(pls), pls if len(pls) < 8 else "")
    if base is None and res["nom"] != "base_fusio" and os.path.exists(os.path.join(RESULTS, "base_fusio.json")):
        base = carrega("base_fusio")
    if base is not None:
        for k in ("acc", "dur", "prov"):
            m, lo, hi, n = compara(base, res, k)
            print(f"   Δ{k} vs {base['nom']}: {m:+.4f} [{lo:+.4f},{hi:+.4f}] {n}/10")
        ms = [res["partits"][x]["acc"] - base["partits"][x]["acc"] for x in sorted(res["partits"]) if x.startswith("metrica")]
        print(f"   Δacc solo IDSSE: {np.mean([res['partits'][x]['acc'] - base['partits'][x]['acc'] for x in res['partits'] if x.startswith('idsse')]):+.4f}"
              f"   solo Metrica: {np.mean(ms):+.4f}")


if __name__ == "__main__":
    import time
    # comprobación: el Viterbi en C y la métrica rápida coinciden con el código de Jan
    e = entrada("idsse", IDSSE[0], 0)
    st = stats("tots")
    cfg = {"w": 0.75, "lam": 0.25, "s_dur": 0.5, "porta": False}
    t0 = time.time()
    ref = dc.descodifica(e["p"], e["janv1"], ~e["clar"], e["period"], st, [cfg])[0]
    t1 = time.time()
    mine = decode_fusio(e, st, cfg)
    t2 = time.time()
    refc = pm.label_codes(ref)
    print("viterbi igual:", bool((refc == mine).all()), f"jan {t1 - t0:.1f}s  C {t2 - t1:.3f}s")
    print("spellF1 pm:", pm.evaluate(e["gt_acc"], refc)["spell_f1_mean"], "rápida:", spell_f1(e["gt_acc"], mine))
    t0 = time.time()
    avalua_familia("base_fusio", decode_fusio, GRAELLA_FUSIO)
    print(f"{time.time() - t0:.0f} s")
