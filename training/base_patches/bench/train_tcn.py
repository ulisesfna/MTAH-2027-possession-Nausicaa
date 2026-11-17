"""Entrenament de la TCN no causal (models/tcn_v7.py) amb els partits externs degradats.

    P=/workspace/shared/venvs/bench/bin/python
    $P bench/train_tcn.py prova                 # 2 èpoques d'un plec, per estimar el temps
    $P bench/train_tcn.py cv                    # els 8 plecs i les probabilitats dels externs
    $P bench/train_tcn.py final                 # 27 partits, mediana de les millors èpoques
    $P bench/train_tcn.py determinisme          # torna a entrenar un plec i compara sha256
    $P bench/train_tcn.py partit                # probabilitats del partit (només agregats)

Variant amb MTAH_TCN=v7|v7b|v7c|v7d (per defecte v7b). Si la degradació de MTAH_DEG no és la de la
variant, `cv` no torna a entrenar: només recalcula les probabilitats fora de plec d'IDSSE.

Dades: les característiques de bench/features.py (features_v7b), etiqueta label_acc. Entrenen IDSSE
(7) i SkillCorner (20), amb les 3 llavors de degradació i l'intercanvi d'equips com a augmentació.
Metrica no hi entra mai. Plecs: un per partit d'IDSSE (entrena amb els altres 26) i un per a
Metrica (entrena amb els 27). Dins de cada plec, un partit d'IDSSE d'entrenament fa de validació
interna per a l'aturada primerenca. Vegeu docs/tcn.md.
"""
import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

AQUI = os.path.dirname(os.path.abspath(__file__))
ARREL = os.path.dirname(AQUI)
sys.path.insert(0, ARREL)
sys.path.insert(0, AQUI)
import features as ft  # noqa: E402
from build_labels import desa_npz  # noqa: E402
from models import tcn_v7  # noqa: E402

ENTRENAMENT = ("idsse", "skillcorner")
LLAVORS = tuple(int(x) for x in os.environ.get("TCN_LLAVORS", "0,1,2").split(","))   # v10: 0-2 = degradación v8.2, 3-5 = degradación de Jan
PLECS_SC = int(os.environ.get("TCN_PLECS_SC", "0"))   # v10: nº de pliegues extra que dejan fuera SkillCorner (probabilidades fuera de pliegue para el stack)
ETIQUETA = "label_acc"

# variants: v7 (C2, degradació v1, 32 característiques), v7b (C2b, degradació v2, 35), v7c (com la
# v7b, amb 14 èpoques com a màxim; descartada) i v7d (C2c: com la v7b, amb la degradació v3). Es tria amb MTAH_TCN; la degradació, amb MTAH_DEG (versio.py).
VARIANTS = {"v7": {"deg": "v1", "epoques_max": 7}, "v7b": {"deg": "v2", "epoques_max": 7},
            "v7c": {"deg": "v2", "epoques_max": 14}, "v7d": {"deg": "v3", "epoques_max": 7},
            "v8": {"deg": "v8", "epoques_max": 7}, "v9": {"deg": "v9", "epoques_max": 7}}
VARIANT = os.environ.get("MTAH_TCN", "v7b")
if VARIANT not in VARIANTS:      # variantes propias (v10...): degradación del mismo nombre
    VARIANTS[VARIANT] = {"deg": VARIANT, "epoques_max": 7}


def rutes(v=None):
    """Rutes d'una variant. Les probabilitats dels externs depenen també de la degradació: si no és
    la de la variant (p. ex. la v7 sobre la v2), van a un directori a part."""
    import versio
    v = v or VARIANT
    if v not in ("v7", "v7b", "v7c", "v7d"):   # variantes propias: todo en local
        loc = os.path.expanduser(f"~/mtah_data/{v}")
        return {"models": loc + "/tcn_models", "probs": loc + "/tcn_probs", "pesos": loc + f"/tcn_{v}_weights.pt",
                "partit": loc + f"/features_match/tcn_{v}_probs.npz"}
    probs = f"/workspace/shared/tcn_{v}_probs"
    if versio.DEG != VARIANTS[v]["deg"]:
        probs += f"_deg{versio.DEG[1:]}"
    return {"models": f"/workspace/shared/tcn_{v}_models", "probs": probs,
            "pesos": os.path.join(ARREL, "models", f"tcn_{v}_weights.pt"),
            "partit": os.path.join(ft.SORTIDA_PARTIT, f"tcn_{v}_probs.npz")}  # ruta protegida


SORTIDA_MODELS = rutes()["models"]
SORTIDA_PROBS = rutes()["probs"]
PESOS_FINALS = rutes()["pesos"]
PROBS_PARTIT = rutes()["partit"]

# fixats per definició, no s'ajusten
PES_FRONTERA = 3.0
RADI_FRONTERA = 12
LR = 1e-3
WEIGHT_DECAY = 1e-4
CLIP = 1.0
LLAVOR = 0
# per pressupost de temps (vegeu docs/tcn.md)
LONGITUD = 1024
LOT = 16
EPOQUES_MAX = VARIANTS[VARIANT]["epoques_max"]
PACIENCIA = 3

CODI = {"H": 0, "A": 1, "D": 2}


def fixa_llavors(s=LLAVOR):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(os.environ.get("TCN_FILS", "2")))


# ---------------------------------------------------------------- dades

def fitxer(font, partit, llavor):
    return os.path.join(ft.SORTIDA_EXT, font, f"{partit}_s{llavor}.npz")


def partits(font):
    return sorted({os.path.basename(f).rsplit("_s", 1)[0] for f in glob.glob(os.path.join(ft.SORTIDA_EXT, font, "*_s0.npz"))})


def objectius(z):
    """Classe per frame (-1 on no compta) i pes de frontera."""
    lab = z[ETIQUETA].astype("<U1")
    y = np.full(len(lab), -1, np.int64)
    for k, v in CODI.items():
        y[lab == k] = v
    y[(z["valid"] != 1) | (z["period"] <= 0)] = -1
    w = np.ones(len(y), np.float32)
    canvi = np.flatnonzero((y[1:] != y[:-1]) & (y[1:] >= 0) & (y[:-1] >= 0)) + 1
    if canvi.size:
        dif = np.zeros(len(y) + 1, np.int32)
        np.add.at(dif, np.maximum(canvi - 1 - RADI_FRONTERA, 0), 1)
        np.add.at(dif, np.minimum(canvi + RADI_FRONTERA + 1, len(y)), -1)
        w[np.cumsum(dif[:-1]) > 0] = PES_FRONTERA
    return y.astype(np.int8), w


_CAU = {}


def carrega(font, partit, llavor):
    k = (font, partit, llavor)
    if k not in _CAU:
        z = np.load(fitxer(font, partit, llavor), allow_pickle=False)
        y, w = objectius(z)
        _CAU[k] = {"X": z["X"].astype(np.float32), "y": y, "w": w, "noms": [str(n) for n in z["noms"]]}
    return _CAU[k]


def permutacio_intercanvi(noms):
    """Índexs i signes per intercanviar local i visitant amb la rotació de 180° corresponent."""
    idx = []
    for n in noms:
        if n.startswith("loc_"):
            idx.append(noms.index("vis_" + n[4:]))
        elif n.startswith("vis_"):
            idx.append(noms.index("loc_" + n[4:]))
        else:
            idx.append(noms.index(n))
    gira = {"pilota_x", "pilota_y", "pilota_ultima_x", "pilota_ultima_y",
            "loc_centre_x", "loc_centre_y", "vis_centre_x", "vis_centre_y"}
    signe = np.array([-1.0 if n in gira else 1.0 for n in noms], np.float32)
    return np.array(idx), signe


def estandarditzacio(conjunt):
    """Mitjana i desviació per característica, amb els frames de les parts dels partits d'entrenament."""
    s = s2 = None
    n = 0
    for font, partit in conjunt:
        for ll in LLAVORS:
            z = np.load(fitxer(font, partit, ll), allow_pickle=False)
            X = z["X"][z["period"] > 0].astype(np.float64)
            s = X.sum(0) if s is None else s + X.sum(0)
            s2 = (X ** 2).sum(0) if s2 is None else s2 + (X ** 2).sum(0)
            n += len(X)
    mu = s / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0))
    sd[sd < 1e-6] = 1.0
    return mu.astype(np.float32), sd.astype(np.float32)


class Lots:
    """Trossos de LONGITUD frames. A cada època, cada partit fa servir una llavor a l'atzar, el
    desplaçament dels trossos també és a l'atzar, i cada tros s'intercanvia amb probabilitat 0,5."""

    def __init__(self, conjunt, mu, sd, rng):
        self.conjunt, self.mu, self.sd, self.rng = list(conjunt), mu, sd, rng
        d = carrega(*self.conjunt[0], 0)
        self.perm, self.signe = permutacio_intercanvi(d["noms"])

    def epoca(self):
        trossos = []
        for font, partit in self.conjunt:
            ll = int(self.rng.randint(len(LLAVORS)))
            d = carrega(font, partit, LLAVORS[ll])
            n = len(d["y"])
            o = int(self.rng.randint(LONGITUD))
            for a in range(o, n - LONGITUD + 1, LONGITUD):
                if (d["y"][a:a + LONGITUD] >= 0).any():
                    trossos.append((d, a, bool(self.rng.rand() < 0.5)))
        ordre = self.rng.permutation(len(trossos))
        for i in range(0, len(ordre), LOT):
            xs, ys, ws = [], [], []
            for j in ordre[i:i + LOT]:
                d, a, inter = trossos[j]
                x = d["X"][a:a + LONGITUD]
                y = d["y"][a:a + LONGITUD].astype(np.int64)
                if inter:
                    x = x[:, self.perm] * self.signe
                    y = np.where(y == 0, 1, np.where(y == 1, 0, y))
                xs.append(((x - self.mu) / self.sd).T)
                ys.append(y)
                ws.append(d["w"][a:a + LONGITUD])
            yield (torch.from_numpy(np.stack(xs).astype(np.float32)), torch.from_numpy(np.stack(ys)),
                   torch.from_numpy(np.stack(ws)))


def perdua(logits, y, w):
    m = (y >= 0).float()
    ce = F.cross_entropy(logits, y.clamp(min=0), reduction="none")
    return (ce * w * m).sum() / (w * m).sum().clamp(min=1.0)


@torch.no_grad()
def perdua_partit(model, mu, sd, d):
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(((d["X"] - mu) / sd).T))[None]
    v = perdua(model(x), torch.from_numpy(d["y"].astype(np.int64))[None], torch.from_numpy(d["w"])[None])
    model.train()
    return float(v)


# ---------------------------------------------------------------- plecs

def plecs():
    """[(nom, avaluats, entrenament, validació interna)]."""
    ids = partits("idsse")
    sc = partits("skillcorner")
    tots = [("idsse", p) for p in ids] + [("skillcorner", p) for p in sc]
    out = []
    for i, p in enumerate(ids):
        val = ("idsse", ids[(i + 1) % len(ids)])
        ent = [k for k in tots if k != ("idsse", p) and k != val]
        out.append((f"idsse_{p}", [("idsse", p)], ent, val))
    val = ("idsse", ids[0])
    out.append(("metrica", [("metrica", p) for p in partits("metrica")], [k for k in tots if k != val], val))
    for j in range(PLECS_SC):   # v10: SkillCorner fuera de pliegue (cada partido en un pliegue; entrena con los 7 IDSSE y el resto de SkillCorner)
        fora = [("skillcorner", p) for p in sc[j::PLECS_SC]]
        out.append((f"sc_{j}", fora, [k for k in tots if k not in fora and k != val], val))
    return out


def entrena(conjunt, val=None, epoques=EPOQUES_MAX, paciencia=PACIENCIA, registre=print):
    """Entrena des de zero. Amb val, atura quan la pèrdua de validació no millora en `paciencia`
    èpoques i torna el millor estat; sense val, fa exactament `epoques` èpoques."""
    fixa_llavors()
    mu, sd = estandarditzacio(conjunt)
    model = tcn_v7.TCN(n_in=len(mu))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    lots = Lots(conjunt, mu, sd, np.random.RandomState(LLAVOR))
    dval = carrega(*val, 0) if val else None
    hist, millor, estat, sense = [], None, None, 0
    model.train()
    for ep in range(1, epoques + 1):
        t0 = time.time()
        tot, n = 0.0, 0
        for x, y, w in lots.epoca():
            opt.zero_grad()
            l = perdua(model(x), y, w)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            tot += float(l.detach())
            n += 1
        h = {"epoca": ep, "perdua_entrenament": tot / max(n, 1), "passos": n, "segons": round(time.time() - t0, 1)}
        if dval is not None:
            h["perdua_validacio"] = perdua_partit(model, mu, sd, dval)
            if millor is None or h["perdua_validacio"] < millor["perdua_validacio"]:
                millor, estat, sense = h, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                sense += 1
        hist.append(h)
        registre(json.dumps(h))
        if dval is not None and sense >= paciencia:
            break
    if dval is None:
        estat, millor = model.state_dict(), hist[-1]
    return {"estat": estat, "mitjana": mu, "desviacio": sd, "epoca_millor": millor["epoca"], "historial": hist,
            "entrenament": [f"{a}/{b}" for a, b in conjunt], "validacio": f"{val[0]}/{val[1]}" if val else None,
            "config": config()}


def config():
    return {"dilatacions": list(tcn_v7.DILATACIONS), "canals": tcn_v7.CANALS, "dropout": tcn_v7.DROPOUT,
            "pes_frontera": PES_FRONTERA, "radi_frontera": RADI_FRONTERA, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "clip": CLIP, "longitud": LONGITUD, "lot": LOT, "epoques_max": EPOQUES_MAX, "paciencia": PACIENCIA,
            "llavor": LLAVOR, "etiqueta": ETIQUETA, "torch": torch.__version__, "variant": VARIANT}


def ruta_plec(nom, v=None):
    return os.path.join(rutes(v)["models"], f"plec_{nom}.pt")


def desa(ck, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ck, path)


def probs_extern(ck_path, font, partit, llavor):
    model, ck = tcn_v7.carrega(ck_path)
    d = carrega(font, partit, llavor)
    return tcn_v7.probabilitats(model, ck, d["X"])


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def probs_extern_sim(ck_path, font, partit, llavor):
    """Promedio simétrico (E17): media de la predicción directa y de la predicción con los equipos intercambiados
    (misma permutación y giro que la aumentación del entrenamiento), deshaciendo el intercambio en la salida."""
    model, ck = tcn_v7.carrega(ck_path)
    d = carrega(font, partit, llavor)
    idx, sg = permutacio_intercanvi(d["noms"])
    pa = tcn_v7.probabilitats(model, ck, d["X"])
    pb = tcn_v7.probabilitats(model, ck, np.ascontiguousarray(d["X"][:, idx] * sg))[:, [1, 0, 2]]
    return ((pa + pb) / 2).astype(np.float32)


def desa_probs(nom, avaluats, dins):
    for font, partit in avaluats:
        for ll in LLAVORS:
            if os.environ.get("TCN_SIM", "0") == "1":
                ps = probs_extern_sim(ruta_plec(nom), font, partit, ll)
                desa_npz(os.path.join(SORTIDA_PROBS + "_sim", font, f"{partit}_s{ll}.npz"),
                         {"probs": ps, "classes": np.array(tcn_v7.CLASSES),
                          "meta": np.array(json.dumps({"plec": nom, "fora_de_plec": not dins, "simetric": True, "sha256": sha(ps)}, sort_keys=True))})
            p = probs_extern(ruta_plec(nom), font, partit, ll)
            desa_npz(os.path.join(SORTIDA_PROBS, font, f"{partit}_s{ll}.npz"),
                     {"probs": p, "classes": np.array(tcn_v7.CLASSES),
                      "meta": np.array(json.dumps({"plec": nom, "fora_de_plec": not dins, "sha256": sha(p)}, sort_keys=True))})


# ---------------------------------------------------------------- comandes

def cmd_prova(a):
    nom, av, ent, val = plecs()[0]
    t0 = time.time()
    ck = entrena(ent, val, epoques=2, paciencia=99)
    s = time.time() - t0
    print(json.dumps({"plec": nom, "segons_2_epoques": round(s, 1),
                      "segons_epoca": [h["segons"] for h in ck["historial"]],
                      "estimacio_cv_hores_per_epoca": round(8 * s / 2 / 3600, 3)}))


def enllaca_fonts_iguals():
    """Reinferència sobre una altra degradació: SkillCorner i Metrica no canvien entre degradacions,
    així que s'enllacen les probabilitats de la degradació de la variant en comptes de recalcular-les."""
    import versio
    orig = f"/workspace/shared/tcn_{VARIANT}_probs"
    for font in ("skillcorner", "metrica"):
        os.makedirs(os.path.join(SORTIDA_PROBS, font), exist_ok=True)
        for f in sorted(glob.glob(os.path.join(orig, font, "*.npz"))):
            desti = os.path.join(SORTIDA_PROBS, font, os.path.basename(f))
            if not os.path.lexists(desti):
                os.symlink(f, desti)
    return versio.DEG


def cmd_cv(a):
    import versio
    reinferencia = versio.DEG != VARIANTS[VARIANT]["deg"]
    if reinferencia:
        enllaca_fonts_iguals()
    resum = {}
    for nom, av, ent, val in plecs():
        if a.plec and nom not in a.plec:
            continue
        path = ruta_plec(nom)
        if reinferencia:
            if not os.path.exists(path):
                sys.exit(f"falta {path}: la reinferència no entrena")
            if nom != "metrica":
                desa_probs(nom, av, dins=False)
            continue
        if not os.path.exists(path) or a.refes:
            t0 = time.time()
            print(f"plec {nom}: {len(ent)} partits d'entrenament, validació {val[0]}/{val[1]}", flush=True)
            ck = entrena(ent, val, registre=lambda s: print("  ", s, flush=True))
            ck["segons"] = round(time.time() - t0, 1)
            desa(ck, path)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        desa_probs(nom, av, dins=False)
        if nom == "metrica" and not PLECS_SC:  # SkillCorner és dins de l'entrenament: només informatiu
            desa_probs(nom, [("skillcorner", p) for p in partits("skillcorner")], dins=True)
        resum[nom] = {"epoca_millor": ck["epoca_millor"], "epoques_fetes": len(ck["historial"]), "segons": ck.get("segons")}
        print(nom, json.dumps(resum[nom]), flush=True)
    if resum:
        with open(os.path.join(SORTIDA_MODELS, "resum_cv.json"), "w") as fh:
            json.dump(resum, fh, indent=1)


def epoques_finals():
    eps = [torch.load(ruta_plec(nom), map_location="cpu", weights_only=False)["epoca_millor"] for nom, *_ in plecs()]
    return int(np.floor(np.median(eps) + 0.5)), eps


def cmd_final(a):
    e, eps = epoques_finals()
    ids, sc = partits("idsse"), partits("skillcorner")
    conjunt = [("idsse", p) for p in ids] + [("skillcorner", p) for p in sc]
    print(f"model final: {len(conjunt)} partits, {e} èpoques (millors per plec: {eps})", flush=True)
    t0 = time.time()
    ck = entrena(conjunt, None, epoques=e, registre=lambda s: print("  ", s, flush=True))
    ck["segons"] = round(time.time() - t0, 1)
    ck["epoques_plecs"] = eps
    ck["estat"] = {k: v.clone() for k, v in ck["estat"].items()}
    desa(ck, PESOS_FINALS)
    print(json.dumps({"pesos": PESOS_FINALS, "mida_MB": round(os.path.getsize(PESOS_FINALS) / 1e6, 2),
                      "segons": ck["segons"]}))


def cmd_determinisme(a):
    nom, av, ent, val = plecs()[0]
    ref = ruta_plec(nom)
    tmp = os.path.join(SORTIDA_MODELS, f"repeticio_{nom}.pt")
    ck = entrena(ent, val, registre=lambda s: print("  ", s, flush=True))
    desa(ck, tmp)
    font, partit = av[0]
    p1, p2 = probs_extern(ref, font, partit, 0), probs_extern(tmp, font, partit, 0)
    # inferència del model final dues vegades sobre un extern
    q1 = probs_extern(PESOS_FINALS, "metrica", partits("metrica")[0], 0)
    q2 = probs_extern(PESOS_FINALS, "metrica", partits("metrica")[0], 0)
    print(json.dumps({"plec": nom, "sha_original": sha(p1), "sha_repeticio": sha(p2), "plec_igual": sha(p1) == sha(p2),
                      "final_sha_1": sha(q1), "final_sha_2": sha(q2), "final_igual": sha(q1) == sha(q2)}, indent=1))


def cmd_partit(a):
    """Probabilitats del model final al partit. Només s'imprimeixen agregats."""
    import adapter
    import games
    sys.path.insert(0, os.path.join(ARREL, "evaluation"))
    import possession_metrics as pm
    from run_evaluation import find_gt
    d = adapter.carrega_partit()
    X, noms, _ = ft.features(d)
    model, ck = tcn_v7.carrega(PESOS_FINALS)
    p = tcn_v7.probabilitats(model, ck, X)
    desa_npz(PROBS_PARTIT, {"probs": p, "classes": np.array(tcn_v7.CLASSES)})
    graded = np.zeros(len(p), bool)
    runs = [(a0, b0 + 1) for _, a0, b0 in d["half_blocks"]]
    for a0, b0 in runs:
        graded[a0:b0] = True
    am = np.array(tcn_v7.CLASSES)[p.argmax(1)]
    codes = pm.label_codes(am)
    st = pm.possession_stats(pm.fill_abstentions(codes, runs), runs)
    gt = pm.label_codes(pm.read_label_csv(find_gt("public")))
    r = pm.evaluate(gt, codes)
    print(json.dumps({"partit": games.GAME, "frames_parts": int(graded.sum()),
                      "prob_mitjana": {c: round(float(p[graded, i].mean()), 4) for i, c in enumerate(tcn_v7.CLASSES)},
                      "pct_argmax": {c: round(100.0 * float((am[graded] == c).mean()), 2) for c in tcn_v7.CLASSES},
                      "possessions_argmax": st["n_spells"], "durada_mitjana_s": round(st["mean_duration_s"], 2),
                      "public_no_s_hi_decideix": {"spellF1": round(r["spell_f1_mean"], 4), "macroF1": round(r["macro_f1"], 4)}},
                     ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ordre", choices=["prova", "cv", "final", "determinisme", "partit"])
    ap.add_argument("--plec", nargs="*")
    ap.add_argument("--refes", action="store_true")
    a = ap.parse_args()
    {"prova": cmd_prova, "cv": cmd_cv, "final": cmd_final, "determinisme": cmd_determinisme, "partit": cmd_partit}[a.ordre](a)


if __name__ == "__main__":
    main()
