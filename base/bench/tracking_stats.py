"""Estadístiques agregades d'un tracking, iguals per al partit i per als externs.

Entrada: arrays en metres a 25 fps i una màscara de frames que compten. Sortida: un dict
d'agregats (taxes, quantils, histogrames, desviacions), mai valors per frame.
"""
import numpy as np

FPS = 25
QUANTILS = [10, 25, 50, 75, 90, 99]
VEL_FRANGES = [(0, 5), (5, 15), (15, np.inf)]  # m/s
ALCADA_MAX = 1.5  # m
# Savitzky-Golay del soroll: finestra de 11 frames (0,44 s) i ordre 2
SG_FINESTRA = 11
SG_ORDRE = 2
LOG_VORES = [2 ** k for k in range(0, 16)]  # franges logarítmiques en frames: [1,2), [2,4), ...

# condicionament v2 de l'inici de forat: línia més propera i velocitat mitjana dels jugadors
LINIA_FRANGES = ("fora", "l<2", "l2-10", "l>10")
JUG_FRANGES = ("j<1.5", "j1.5-3", "j>3")
VEL_CLAUS = ("v<5", "v5-15", "v>15", "desconeguda")
JUG_DESPL = 25   # velocitat dels jugadors: desplaçament en 1 s, per no dependre del soroll
MIN_FORATS = 10  # una franja amb menys forats s'ajunta (vegeu agrupa_v2)


def trams(m):
    """(inici, fi) dels trams de True."""
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def distribucio(llargades_frames):
    a = np.asarray(llargades_frames, float)
    if a.size == 0:
        return {"n": 0}
    hist, _ = np.histogram(a, bins=LOG_VORES + [np.inf])
    return {
        "n": int(a.size),
        "mitjana_s": round(float(a.mean()) / FPS, 4),
        "quantils_s": {f"p{q}": round(float(np.percentile(a, q)) / FPS, 4) for q in QUANTILS},
        "hist_log_frames": {"vores": LOG_VORES, "recomptes": hist.tolist()},
        # funció de quantils (0..100) en frames, per mostrejar
        "quantils101_frames": [round(float(x), 3) for x in np.percentile(a, np.arange(101))],
    }


def franja_vel(v):
    for i, (a, b) in enumerate(VEL_FRANGES):
        if a <= v < b:
            return i
    return -1


def clau_condicio(vel, z):
    k = "desconeguda" if not np.isfinite(vel) else ["v<5", "v5-15", "v>15"][franja_vel(vel)]
    if z is not None:
        k += "|" + ("z?" if not np.isfinite(z) else ("z<=1.5" if z <= ALCADA_MAX else "z>1.5"))
    return k


def franja_linia(x, y, camp):
    """Distància a la línia més propera (banda o fons), en franges. x, y amb origen al centre."""
    L, W = camp
    d = np.minimum(L / 2 - np.abs(x), W / 2 - np.abs(y))
    return np.select([d < 0, d < 2, d <= 10], [0, 1, 2], 3)


def vel_jugadors(xy):
    """Velocitat mitjana (m/s) dels jugadors visibles a cada frame: desplaçament del mateix slot en
    JUG_DESPL frames. NaN si cap slot no té els dos extrems."""
    n = len(xy)
    v = np.full(xy.shape[:2], np.nan)
    k = JUG_DESPL
    v[k:] = np.hypot(xy[k:, :, 0] - xy[:-k, :, 0], xy[k:, :, 1] - xy[:-k, :, 1]) * FPS / k
    ok = np.isfinite(v)
    with np.errstate(invalid="ignore"):
        out = np.where(ok.any(1), np.nansum(np.where(ok, v, 0), 1) / np.maximum(ok.sum(1), 1), np.nan)
    return out[:n]


def franja_jug(v):
    """Franja de velocitat mitjana dels jugadors; sense dada compta com a la del mig."""
    return np.select([~np.isfinite(v), v < 1.5, v <= 3], [1, 0, 1], 2)


def stats_pilota_v2(bola, xy, camp, mask, vel_t, t, inici, llarg):
    """Inicis de forat per (velocitat de la pilota, línia, jugadors) i durades per jugadors."""
    vj = vel_jugadors(xy)
    li = franja_linia(bola[t, 0], bola[t, 1], camp)
    ju = franja_jug(vj[t])
    ve = np.array([VEL_CLAUS.index(clau_condicio(v, None)) for v in vel_t])
    cel = {}
    for a in range(len(VEL_CLAUS)):
        for b in range(len(LINIA_FRANGES)):
            for c in range(len(JUG_FRANGES)):
                m = (ve == a) & (li == b) & (ju == c)
                cel[f"{VEL_CLAUS[a]}|{LINIA_FRANGES[b]}|{JUG_FRANGES[c]}"] = {
                    "oportunitats": int(m.sum()), "inicis": int((m & inici).sum())}
    durs, crues = {}, {}
    for c in range(len(JUG_FRANGES)):
        d = llarg[t[(ju == c) & inici] + 1]
        crues[JUG_FRANGES[c]] = d[d > 0]
        durs[JUG_FRANGES[c]] = distribucio(crues[JUG_FRANGES[c]])
    out = {"cel": cel, "durades_jugadors": durs,
           "vel_jugadors_quantils": {f"p{q}": round(float(np.nanpercentile(vj[mask], q)), 3) for q in (10, 50, 90)}
           if np.isfinite(vj[mask]).any() else None}
    taxes, grups = agrupa_v2(out)
    out["taxes_agrupades"] = taxes
    out["durades_agrupades"] = {j: {"grup": g["grup"], **distribucio(np.concatenate([crues[x] for x in g["grup"]]))}
                                for j, g in grups.items()}
    return out


def nivells_v2(k):
    """Grups de la regla jeràrquica per a la cel·la k ("vel|línia|jugadors"), de més fi a més gruixut."""
    v, l, j = k.split("|")
    li = LINIA_FRANGES.index(l)
    veina = {0: 1, 1: 2, 2: 3, 3: 2}
    lenta = v == "v<5"
    vv = lambda a: (a == "v<5") == lenta
    return [("cel·la", lambda a, b, c: (a, b, c) == (v, l, j)),
            ("vel2+linia+jugadors", lambda a, b, c: vv(a) and (b, c) == (l, j)),
            ("vel2+linia", lambda a, b, c: vv(a) and b == l),
            ("vel2+linia+veina", lambda a, b, c: vv(a) and b in (l, LINIA_FRANGES[veina[li]])),
            ("vel2", lambda a, b, c: vv(a)),
            ("tot", lambda a, b, c: True)]


def stats_estat(bola, xy, camp, mask, morta):
    """v3: inicis de forat per cel·la v2 separant la pilota viva de la morta, i durades per estat.

    mask: frames que compten (a SkillCorner, vàlids i dins de les parts). morta: label_acc == D.
    L'estat és el del frame t, l'últim amb pilota abans del forat."""
    present = np.isfinite(bola[:, 0]) & mask
    absent = ~np.isfinite(bola[:, 0]) & mask
    t = np.flatnonzero(present[:-1] & mask[1:])
    inici = ~np.isfinite(bola[t + 1, 0])
    vel = np.full(t.size, np.nan)
    ok = t > 0
    ok[ok] = present[t[ok] - 1]
    vel[ok] = np.hypot(bola[t[ok], 0] - bola[t[ok] - 1, 0], bola[t[ok], 1] - bola[t[ok] - 1, 1]) * FPS
    vj = vel_jugadors(xy)
    li = franja_linia(bola[t, 0], bola[t, 1], camp)
    ju = franja_jug(vj[t])
    ve = np.array([VEL_CLAUS.index(clau_condicio(x, None)) for x in vel])
    es = np.asarray(morta, bool)[t]
    llarg = np.zeros(len(bola), int)
    for a, b in trams(absent):
        llarg[a] = b - a
    cel = {}
    for a in range(len(VEL_CLAUS)):
        for b in range(len(LINIA_FRANGES)):
            for c in range(len(JUG_FRANGES)):
                m = (ve == a) & (li == b) & (ju == c)
                cel[f"{VEL_CLAUS[a]}|{LINIA_FRANGES[b]}|{JUG_FRANGES[c]}"] = {
                    e: {"oportunitats": int((m & (es == (e == "D"))).sum()),
                        "inicis": int((m & inici & (es == (e == "D"))).sum())} for e in ("viva", "D")}
    durs = {}
    for e in ("viva", "D"):
        d = llarg[t[inici & (es == (e == "D"))] + 1]
        durs[e] = d[d > 0]
    return cel, durs


def agrupa_estat(cel):
    """r = taxa(D) / taxa(viva) per cel·la, amb la regla de agrupa_v2: el primer grup que tingui
    MIN_FORATS inicis o més en tots dos estats."""
    out = {}
    for k in cel:
        for nom, f in nivells_v2(k):
            ks = [x for x in cel if f(*x.split("|"))]
            s = {e: {q: sum(cel[x][e][q] for x in ks) for q in ("inicis", "oportunitats")} for e in ("viva", "D")}
            if min(s[e]["inicis"] for e in s) >= MIN_FORATS or nom == "tot":
                tx = {e: s[e]["inicis"] / s[e]["oportunitats"] if s[e]["oportunitats"] else 0.0 for e in s}
                out[k] = {"grup": nom, **{f"{q}_{e}": s[e][q] for e in s for q in ("inicis", "oportunitats")},
                          "taxa_viva": tx["viva"], "taxa_D": tx["D"],
                          "r": tx["D"] / tx["viva"] if tx["viva"] > 0 else None}
                break
    return out


def agrupa_v2(v2):
    """Regla jeràrquica fixa per a les franges amb menys de MIN_FORATS forats. Una cel·la
    (velocitat, línia, jugadors) fa servir el primer d'aquests grups que tingui prou inicis:
    (1) la cel·la; (2) la velocitat de la pilota en dues franges (lenta: v<5; ràpida: la resta,
    inclosa la desconeguda), amb la mateixa línia i jugadors; (3) a més, totes les franges de
    jugadors; (4) a més, la línia veïna cap a dins del camp (fora -> l<2, l<2 -> l2-10,
    l2-10 -> l>10, l>10 -> l2-10); (5) només la velocitat en dues franges; (6) tot.
    Durades per jugadors: una franja amb pocs forats s'ajunta amb la del mig (la del mig, amb la
    més poblada de les altres dues), i si encara no n'hi ha prou, totes.
    Torna {cel·la: {taxa, grup, inicis, oportunitats}} i {franja: {grup}}."""
    cel = v2["cel"]

    def suma(f):
        ks = [k for k in cel if f(*k.split("|"))]
        return ks, sum(cel[k]["inicis"] for k in ks), sum(cel[k]["oportunitats"] for k in ks)

    taxes = {}
    for k in cel:
        for nom, f in nivells_v2(k):
            ks, i, o = suma(f)
            if i >= MIN_FORATS or nom == "tot":
                taxes[k] = {"grup": nom, "inicis": i, "oportunitats": o, "taxa": i / o if o else 0.0}
                break
    durs = v2["durades_jugadors"]
    n = {j: durs[j].get("n", 0) for j in JUG_FRANGES}
    dur = {}
    for j in JUG_FRANGES:
        if n[j] >= MIN_FORATS:
            grup = [j]
        else:
            if j == JUG_FRANGES[1]:
                altra = max((JUG_FRANGES[0], JUG_FRANGES[2]), key=lambda x: n[x])
                grup = [j, altra]
            else:
                grup = [j, JUG_FRANGES[1]]
            if sum(n[x] for x in grup) < MIN_FORATS:
                grup = list(JUG_FRANGES)
        dur[j] = {"grup": grup}
    return taxes, dur


def stats_pilota(bola, z, mask, xy=None, camp=None):
    present = np.isfinite(bola[:, 0]) & mask
    absent = (~np.isfinite(bola[:, 0])) & mask
    n = int(mask.sum())
    forats = [b - a for a, b in trams(absent)]
    # taxa d'inici de forat: oportunitat = frame t amb pilota, t+1 dins de la màscara
    t = np.flatnonzero(present[:-1] & mask[1:])
    inici = ~np.isfinite(bola[t + 1, 0])
    vel = np.full(t.size, np.nan)
    ok = t > 0
    ok[ok] = present[t[ok] - 1]
    tt = t[ok]
    vel[ok] = np.hypot(bola[tt, 0] - bola[tt - 1, 0], bola[tt, 1] - bola[tt - 1, 1]) * FPS
    zz = None if z is None else z[t]
    claus = np.array([clau_condicio(v, None if zz is None else zz[i]) for i, v in enumerate(vel)])
    # durada de cada forat segons la condició del frame anterior
    llarg = np.zeros(len(bola), int)
    for a, b in trams(absent):
        llarg[a] = b - a
    cond = {}
    for k in sorted(set(claus.tolist())):
        m = claus == k
        durs = llarg[t[m & inici] + 1]
        cond[k] = {"oportunitats": int(m.sum()), "inicis": int((m & inici).sum()),
                   "taxa_inici": round(float((m & inici).sum()) / max(1, int(m.sum())), 6),
                   "durades": distribucio(durs[durs > 0])}
    out = {"pct_absent": round(100.0 * absent.sum() / max(1, n), 3), "forats": distribucio(forats),
           "forats_per_min": round(len(forats) / (n / FPS / 60), 3) if n else None,
           "inici_condicionat": cond, "te_alcada": z is not None}
    if xy is not None and camp is not None:
        out["inici_v2"] = stats_pilota_v2(bola, xy, camp, mask, vel, t, inici, llarg)
    return out


def stats_jugadors(xy, equip, mask):
    pres = np.isfinite(xy[..., 0]) & mask[:, None]
    vis = {str(e): pres[:, equip == e].sum(axis=1)[mask] for e in (0, 1)}
    frag, buits = [], []
    for s in range(xy.shape[1]):
        tr = trams(pres[:, s])
        frag += [b - a for a, b in tr]
        buits += [tr[i + 1][0] - tr[i][1] for i in range(len(tr) - 1)]
    sense_equip = pres[:, ~np.isin(equip, (0, 1))].sum()
    n = int(mask.sum())
    return {
        "visibles_per_equip": {e: {"mitjana": round(float(v.mean()), 3),
                                   "quantils": {f"p{q}": float(np.percentile(v, q)) for q in (10, 50, 90)},
                                   "hist": np.bincount(v, minlength=16).tolist()} for e, v in vis.items()},
        "visibles_mitjana": round(float(np.mean([v.mean() for v in vis.values()])), 3),
        "fragments": distribucio(frag),
        "fragments_per_min": round(len(frag) / (n / FPS / 60), 3) if n else None,
        "buits_entre_fragments_slot": distribucio(buits),
        "n_slots": int(xy.shape[1]),
        "pct_deteccions_sense_equip": round(100.0 * float(sense_equip) / max(1, int(pres.sum())), 3),
    }


def _coef_sg(w=SG_FINESTRA, o=SG_ORDRE):
    k = np.arange(w) - w // 2
    a = np.vander(k, o + 1, increasing=True)
    return np.linalg.pinv(a)[0]  # valor suavitzat al centre


def residu_sg(serie_xy, pres):
    """Residus (posició - suavitzada) als punts centrals de cada tram continu prou llarg."""
    c = _coef_sg()
    out = []
    for a, b in trams(pres):
        if b - a < SG_FINESTRA:
            continue
        for d in (0, 1):
            x = serie_xy[a:b, d].astype(float)
            s = np.convolve(x, c[::-1], mode="valid")
            out.append(x[SG_FINESTRA // 2: SG_FINESTRA // 2 + s.size] - s)
    return np.concatenate(out) if out else np.zeros(0)


def stats_soroll(bola, xy, mask):
    rb = residu_sg(bola[:, :2], np.isfinite(bola[:, 0]) & mask)
    rj = [residu_sg(xy[:, s], np.isfinite(xy[:, s, 0]) & mask) for s in range(xy.shape[1])]
    rj = np.concatenate(rj) if rj else np.zeros(0)
    return {"finestra_sg": SG_FINESTRA, "ordre_sg": SG_ORDRE,
            "pilota_std_m": round(float(rb.std()), 5) if rb.size else None,
            "jugadors_std_m": round(float(rj.std()), 5) if rj.size else None,
            "n_residus_pilota": int(rb.size), "n_residus_jugadors": int(rj.size)}


def estadistiques(bola_xy, jugadors_xy, equip, mask, bola_z=None, camp=None):
    """bola_xy (n,2), jugadors_xy (n,S,2) en metres amb origen al centre; equip (S,) amb 0/1 (altres =
    sense equip). Amb camp=(L, W) s'afegeix el condicionament v2 dels forats."""
    mask = np.asarray(mask, bool)
    return {"frames": int(mask.sum()), "minuts": round(float(mask.sum()) / FPS / 60, 2),
            "pilota": stats_pilota(bola_xy, bola_z, mask, jugadors_xy, camp),
            "jugadors": stats_jugadors(jugadors_xy, np.asarray(equip), mask),
            "soroll": stats_soroll(bola_xy, jugadors_xy, mask)}


def resum(s):
    """Línia curta d'agregats per imprimir."""
    p, j, r = s["pilota"], s["jugadors"], s["soroll"]
    taxes = {k: v["taxa_inici"] for k, v in p["inici_condicionat"].items()}
    return {"minuts": s["minuts"], "pct_pilota_absent": p["pct_absent"],
            "forat_mediana_s": p["forats"].get("quantils_s", {}).get("p50"),
            "forats_quantils_s": p["forats"].get("quantils_s"), "taxes_inici": taxes,
            "fragment_mediana_s": j["fragments"].get("quantils_s", {}).get("p50"),
            "visibles_per_equip": {e: v["mitjana"] for e, v in j["visibles_per_equip"].items()},
            "pct_sense_equip": j["pct_deteccions_sense_equip"],
            "soroll_jugadors_m": r["jugadors_std_m"], "soroll_pilota_m": r["pilota_std_m"],
            "inici_v2": resum_v2(p.get("inici_v2"))}


def marginals_v2(v2):
    """Taxa d'inici per franja de cada senyal (sumant les altres dues)."""
    out = {}
    for pos, franges in ((0, VEL_CLAUS), (1, LINIA_FRANGES), (2, JUG_FRANGES)):
        for f in franges:
            ks = [k for k in v2["cel"] if k.split("|")[pos] == f]
            i = sum(v2["cel"][k]["inicis"] for k in ks)
            o = sum(v2["cel"][k]["oportunitats"] for k in ks)
            out[f] = {"inicis": i, "oportunitats": o, "taxa": round(i / o, 6) if o else None}
    return out


def resum_v2(v2):
    if not v2:
        return None
    return {"marginals": marginals_v2(v2),
            "grups": {k: v["grup"] for k, v in v2["taxes_agrupades"].items() if v["oportunitats"]},
            "durades_mediana_s": {j: v.get("quantils_s", {}).get("p50") for j, v in v2["durades_agrupades"].items()},
            "vel_jugadors_quantils": v2["vel_jugadors_quantils"]}
