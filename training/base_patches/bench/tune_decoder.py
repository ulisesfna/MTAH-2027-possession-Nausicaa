"""Ajust niat del descodificador (models/decoder.py) amb els partits d'IDSSE.

    P=/workspace/shared/venvs/bench/bin/python
    $P bench/tune_decoder.py entrades     # janv1_notag i el seu control clar per a cada extern
    $P bench/tune_decoder.py graella      # spellF1 de tota la graella als partits d'ajust de cada plec
    $P bench/tune_decoder.py tria         # paràmetres triats per plec i configuració

Plecs: per a cada partit d'IDSSE que es deixa fora, les estadístiques de l'HMM surten dels altres
26 partits d'entrenament (IDSSE i SkillCorner) i els paràmetres es trien amb els altres 6 d'IDSSE.
Per a Metrica (i per al partit), les estadístiques surten dels 27 i els paràmetres, dels 7 d'IDSSE.
Les probabilitats de la TCN són sempre les fora de plec de la variant (train_tcn.rutes). La
graella es calcula per a cada variant de la TCN (--variants); janv1_vit no en depèn.
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
ARREL = os.path.dirname(AQUI)
sys.path.insert(0, ARREL)
sys.path.insert(0, AQUI)
import versio  # noqa: E402
from build_labels import desa_npz  # noqa: E402
from models import decoder as dc  # noqa: E402

DADES = versio.DADES
FEATS = versio.FEATS
SORTIDA = {"v1": "/workspace/shared/decoder_v7", "v2": "/workspace/shared/decoder_v7b",
           "v3": "/workspace/shared/decoder_v7d"}.get(versio.DEG, versio.LOCAL + "/decoder")
LLAVORS = (0, 1, 2)
ENTRENAMENT = ("idsse", "skillcorner")

# graella fixa; l'ordre dins de cada configuració és el de desempat (el primer guanya).
# C2b la va ampliar a les vores; la de C3 era w ∈ {0,25; 0,5; 0,75} i lam ∈ {0,5; 1; 2}.
GRAELLA_W = (0.5, 0.75, 1.0)
GRAELLA_LAM = (0.25, 0.5, 1.0)
GRAELLA_SDUR = (0.5, 1.0)
GRAELLA_FACTOR = (0.0, 0.15, 0.25, 0.35)
CONFIGS = ("janv1_vit", "porta", "fusio", "tcn_vit")  # de més simple a més complexa
# "vora" (v7): lam = 0,25 quedava a la vora, i es baixa només per a porta i tcn_vit
# "c2c" (v7d): porta, fusio i tcn_vit amb lam encara més baix
# "v9" i "v9r": fusio amb modulació dinàmica w_t = w * (1 - factor * conf_total); v9r afegeix TCN neutra en forats
GRAELLES = {"c2b": {"lam": GRAELLA_LAM, "configs": CONFIGS},
            "vora": {"lam": (0.1, 0.15, 0.25), "configs": ("porta", "tcn_vit")},
            "c2c": {"lam": (0.05, 0.1, 0.15, 0.25), "configs": ("porta", "fusio", "tcn_vit")},
            "v9": {"lam": (0.05, 0.1, 0.15, 0.25), "configs": ("fusio",)},
            "v9r": {"lam": (0.05, 0.1, 0.15, 0.25), "configs": ("fusio",)}}


def graella(nom, g="c2b"):
    ws = {"janv1_vit": (0.0,), "tcn_vit": (1.0,)}.get(nom, GRAELLA_W)
    if g in ("v9", "v9r"):
        es_r = g == "v9r"
        return [{"w": w, "lam": l, "s_dur": s, "factor": f, "r": es_r, "porta": False}
                for f, w, l, s in itertools.product(GRAELLA_FACTOR, ws, GRAELLES[g]["lam"], GRAELLA_SDUR)]
    return [{"w": w, "lam": l, "s_dur": s, "porta": nom == "porta"}
            for w, l, s in itertools.product(ws, GRAELLES[g]["lam"], GRAELLA_SDUR)]


def tota_la_graella(g="c2b"):
    return [(nom, c) for nom in GRAELLES[g]["configs"] for c in graella(nom, g)]


def clau_cfg(nom, c):
    f_str = f"|f={c['factor']}" if "factor" in c else ""
    r_str = "|r" if c.get("r") else ""
    return f"{nom}|w={c['w']}|lam={c['lam']}|s={c['s_dur']}{f_str}{r_str}"


def partits(font):
    return sorted({os.path.basename(f).rsplit("_s", 1)[0] for f in glob.glob(os.path.join(DADES, font, "*_s0.npz"))})


# ---------------------------------------------------------------- entrades per extern

def janv1_amb_control(d):
    """Etiquetes de janv1_notag i el seu control clar (memoritzat dins de d)."""
    if "_janv1" not in d:
        from starter import principled_v1 as pv
        cfg = {**pv.CONFIGS["p3_notag"], "exposa_control": True}
        kin = pv.kinematics_from_arrays(d["px"].copy(), d["py"].copy(), d["mean_speed"].copy(),
                                        d["bx"].copy(), d["by"].copy(), d["team_of_track"], d["pitch"])
        st = {"home": None, "away": None, "set_pieces": None}
        labels, info = pv.predict(kin, st, d["n_frames"], d["half_blocks"], d["kickoff_2_frame"], d["pitch"], cfg)
        d["_kin"] = kin
        d["_janv1"] = (labels, info["control_clar"])
    return d["_janv1"]


def conf_fisica(d):
    """Confiança física cinemàtica invariant (kappa <= 0.95 adimensional i exp(-0.20 * rel_speed)).

    Nota: principled_v1 interpola forats de pilota <= 25 frames. En forats curts, phys_conf es
    calcula sobre la pilota interpolada i pot ser > 0; en forats llargs kappa es 1e9 i phys_conf = 0.
    """
    janv1_amb_control(d)
    kin = d["_kin"]
    h_eff = kin["home_min_kappa"]
    a_eff = kin["away_min_kappa"]
    h_rel = np.nan_to_num(kin["home_rel_speed"], nan=np.inf)
    a_rel = np.nan_to_num(kin["away_rel_speed"], nan=np.inf)
    h_match = np.exp(-0.20 * h_rel) * ((~np.isnan(h_eff)) & (h_eff <= 0.95))
    a_match = np.exp(-0.20 * a_rel) * ((~np.isnan(a_eff)) & (a_eff <= 0.95))
    return np.maximum(h_match, a_match)


def ruta_entrada(font, partit, ll):
    return os.path.join(SORTIDA, "entrades", font, f"{partit}_s{ll}.npz")


def _entrada(path):
    import adapter
    import harness
    d = adapter.carrega_extern(path)
    labels, clar = janv1_amb_control(d)
    phys_conf = conf_fisica(d)
    r = np.isfinite(d["bx"]) & np.isfinite(d["by"])
    g = {e: harness.gt_avaluable(d["gt"], e)[0] for e in ("label_acc", "label_dur")}
    desa_npz(ruta_entrada(d["font"], d["partit"], d["llavor"]),
             {"janv1": labels, "clar": clar, "phys_conf": phys_conf, "r": r,
              "period": d["period"], "gt_acc": g["label_acc"], "gt_dur": g["label_dur"]})
    return path


def entrada(font, partit, ll, v):
    import train_tcn as tt
    z = np.load(ruta_entrada(font, partit, ll), allow_pickle=False)
    p = np.load(os.path.join(tt.rutes(v)["probs"], font, f"{partit}_s{ll}.npz"), allow_pickle=False)["probs"]
    return {k: z[k] for k in z.files} | {"p": p}


# ---------------------------------------------------------------- plecs i estadístiques

def codis_entrenament(font, partit):
    z = np.load(os.path.join(FEATS, font, f"{partit}_s0.npz"), allow_pickle=False)
    lab = z["label_acc"].astype("<U1")
    y = np.full(len(lab), -1, np.int64)
    for i, c in enumerate(dc.CLASSES):
        y[lab == c] = i
    y[z["valid"] != 1] = -1
    return y, z["period"]


def plecs():
    """{nom: (partits avaluats, partits d'entrenament de l'HMM, partits d'ajust)}."""
    ids, sc = partits("idsse"), partits("skillcorner")
    tots = [("idsse", p) for p in ids] + [("skillcorner", p) for p in sc]
    out = {f"idsse_{p}": ([("idsse", p)], [k for k in tots if k != ("idsse", p)], [("idsse", q) for q in ids if q != p])
           for p in ids}
    out["tots"] = ([("metrica", p) for p in partits("metrica")], tots, [("idsse", q) for q in ids])
    return out


def plec_per_entrenament(claus):
    for nom, (av, ent, aj) in plecs().items():
        if set(ent) == set(claus):
            return nom
    raise KeyError("cap plec del descodificador no entrena amb aquests partits")


def estadistiques_plec(nom):
    path = os.path.join(SORTIDA, f"estadistiques_{nom}.json")
    if not os.path.exists(path):
        st = dc.estadistiques([codis_entrenament(*k) for k in plecs()[nom][1]])
        os.makedirs(SORTIDA, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(st, fh, indent=1)
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- graella

def _puntua(args):
    import harness
    nom, font, partit, ll, v, g = args
    st = estadistiques_plec(nom)
    e = entrada(font, partit, ll, v)
    cfgs = tota_la_graella(g)
    if g in ("v9", "v9r"):
        conf = e["phys_conf"]
        r = e["r"]
    else:
        conf = e.get("phys_conf")
        r = e.get("r")
    out = dc.descodifica(e["p"], e["janv1"], ~e["clar"], e["period"], st, [c for _, c in cfgs], conf=conf, r=r)
    return nom, v, f"{font}/{partit}", ll, {clau_cfg(n, c): harness.avalua(e["gt_acc"], o)["spellF1"] for (n, c), o in zip(cfgs, out)}


def ruta_taula(nom, v, g="c2b"):
    return os.path.join(SORTIDA, f"graella_{nom}_{v}{'' if g == 'c2b' else '_' + g}.json")


def cmd_graella(a):
    feina = [(nom, f, p, ll, v, a.graella) for v in a.variants for nom, (_, _, aj) in plecs().items() for f, p in aj
             for ll in LLAVORS if a.refes or not os.path.exists(ruta_taula(nom, v, a.graella))]
    for nom in plecs():
        estadistiques_plec(nom)
    t0 = time.time()
    taules = {}
    with Pool(a.jobs) as pool:
        for nom, v, k, ll, sc in pool.imap(_puntua, feina):
            taules.setdefault((nom, v), {}).setdefault(k, {})[ll] = sc
            print(nom, v, k, ll, f"{time.time() - t0:.0f} s", flush=True)
    for (nom, v), t in taules.items():
        with open(ruta_taula(nom, v, a.graella), "w") as fh:
            json.dump(t, fh, indent=1, sort_keys=True)


def tria(config, nom_plec, v, g="c2b"):
    """Paràmetres de la configuració que maximitzen el spellF1 mitjà dels partits d'ajust del plec."""
    with open(ruta_taula(nom_plec, v, g)) as fh:
        t = json.load(fh)
    millor, valor = None, -1.0
    for c in graella(config, g):
        kc = clau_cfg(config, c)
        x = float(np.mean([np.mean([t[k][ll][kc] for ll in sorted(t[k])]) for k in sorted(t)]))
        if x > valor + 1e-12:
            millor, valor = c, x
    return millor, valor


def ajusta(config, claus_entrenament, v, g="c2b"):
    nom = plec_per_entrenament(claus_entrenament)
    c, val = tria(config, nom, v, g)
    return {"plec": nom, "cfg": c, "spellF1_ajust": val, "st": estadistiques_plec(nom), "variant": v, "graella": g}


def cmd_tria(a):
    sufix = "" if a.graella == "c2b" else "_" + a.graella
    res = {(f"{cfg}_{v}" if cfg != "janv1_vit" else cfg) + sufix:
           {nom: dict(zip(("cfg", "spellF1_ajust"), tria(cfg, nom, v, a.graella))) for nom in plecs()}
           for v in a.variants for cfg in GRAELLES[a.graella]["configs"]}
    print(json.dumps(res, indent=1))
    path = os.path.join(AQUI, "results", f"decoder_parametres_per_plec_deg{versio.DEG[1:]}_{'-'.join(a.variants)}{sufix}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)


def cmd_entrades(a):
    fitxers = sorted(glob.glob(os.path.join(DADES, "*", "*.npz")))
    with Pool(a.jobs) as pool:
        for i, f in enumerate(pool.imap(_entrada, fitxers)):
            print(i, os.path.relpath(f, DADES), flush=True)


def cmd_params(a):
    import subprocess
    porcelain = subprocess.check_output(["git", "status", "--porcelain", "-uno"], cwd=ARREL, text=True).strip()
    if porcelain:
        print("Error: l'arbre git no és net:", file=sys.stderr)
        print(porcelain, file=sys.stderr)
        sys.exit(1)
    if a.graella not in ("v9", "v9r"):
        print(f"Error: --graella ha de ser 'v9' o 'v9r', rebut '{a.graella}'.", file=sys.stderr)
        sys.exit(1)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ARREL, text=True).strip()
    graella_nom = a.graella
    data = ajusta("fusio", set(plecs()["tots"][1]), "v7d", graella_nom)
    res = {
        "_meta": {
            "ordre": f"python bench/tune_decoder.py params --graella {graella_nom}",
            "commit_origen": commit,
            "data": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        **data,
    }
    dest = os.path.join(ARREL, "models", f"markius_{graella_nom}_params.json")
    with open(dest, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(f"Desat {dest} amb èxit.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ordre", choices=["entrades", "graella", "tria", "params"])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--refes", action="store_true")
    ap.add_argument("--variants", nargs="+", default=["v7", "v7b"])
    ap.add_argument("--graella", default="c2b", choices=sorted(GRAELLES))
    a = ap.parse_args()
    {"entrades": cmd_entrades, "graella": cmd_graella, "tria": cmd_tria, "params": cmd_params}[a.ordre](a)


if __name__ == "__main__":
    main()
