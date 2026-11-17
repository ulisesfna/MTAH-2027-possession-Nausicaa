"""Etiquetes H/A/D "tipus GT" per als partits externs (IDSSE, SkillCorner, Metrica).

Llegeix /workspace/shared/external_raw/ i desa un .npz per partit a --out/<font>/<partit>.npz.
Les regles són a bench/label_config.yaml i l'especificació a docs/etiquetes_externes.md.

    /workspace/shared/venvs/bench/bin/python bench/build_labels.py --src all --out /workspace/shared/bench_v7
"""
import argparse
import csv
import glob
import hashlib
import io
import json
import os
import re
import zipfile
from bisect import bisect_left, bisect_right
from datetime import datetime
from multiprocessing import Pool
from xml.etree.ElementTree import iterparse

import numpy as np
import yaml

AQUI = os.path.dirname(os.path.abspath(__file__))
RAW = "/workspace/shared/external_raw"
X_, H_, A_, D_ = 0, 1, 2, 3
LLETRES = np.array(["X", "H", "A", "D"])


def carrega_config(path):
    with open(path, "rb") as fh:
        brut = fh.read()
    cfg = yaml.safe_load(brut)
    cfg["_sha256"] = hashlib.sha256(brut).hexdigest()
    return cfg


def desa_npz(path, arrays):
    """Com np.savez_compressed però amb dates fixes al zip, perquè el fitxer sigui reproduïble."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for k in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[k]), allow_pickle=False)
            info = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, buf.getvalue())
    os.replace(tmp, path)


def trams(v):
    v = np.asarray(v)
    if v.size == 0:
        return []
    talls = np.flatnonzero(v[1:] != v[:-1]) + 1
    ini = np.concatenate([[0], talls])
    fi = np.concatenate([talls, [v.size]])
    return [(int(v[i]), int(i), int(f)) for i, f in zip(ini, fi)]


# ---------------------------------------------------------------- regla de duels

class Accions:
    """Accions controlades per equip, com a intervals [ini, fi] en frames de la font."""

    def __init__(self, llista):
        self.per_equip = {}
        for ini, fi, eq in sorted(llista):
            self.per_equip.setdefault(eq, []).append((ini, fi))
        self.inicis = {eq: [a for a, _ in v] for eq, v in self.per_equip.items()}
        self.llarg = {eq: max((b - a for a, b in v), default=0) for eq, v in self.per_equip.items()}

    def n(self):
        return sum(len(v) for v in self.per_equip.values())

    def hi_ha(self, eq, a, b, marge):
        # alguna acció d'eq que toqui [a - marge, b - 1 + marge]
        if eq not in self.per_equip:
            return False
        ini = self.inicis[eq]
        lo = bisect_left(ini, a - marge - self.llarg[eq])
        hi = bisect_right(ini, b - 1 + marge)
        return any(self.per_equip[eq][i][1] >= a - marge for i in range(lo, hi))


def ratxa_max(m):
    """Llargada de la ratxa més llarga de True."""
    if not m.any():
        return 0
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def aplica_regla(morta, prov, periode, accions, mode, marge, durada_min, acc_amb_durada=False, control=None):
    """Etiqueta per frame (H_, A_, D_, X_). mode: 'acc', 'dur' o 'cru' (accepta tots els canvis).

    acc_amb_durada: a 'acc', el criteri de durada també val com a control.
    control: {equip: màscara per frame}; si hi és, el criteri de durada demana durada_min frames
    seguits de la màscara dins del tram, i no només la llargada del tram."""

    def per_durada(eq, ra, rb):
        if control is None:
            return rb - ra >= durada_min
        return ratxa_max(control[eq][ra:rb]) >= durada_min

    n = len(prov)
    out = np.zeros(n, np.int8)
    out[morta & (periode > 0)] = D_
    for p in sorted(set(periode[periode > 0].tolist())):
        idx = np.flatnonzero(periode == p)
        p0 = int(idx[0])
        actual = 0
        for estat, a, b in trams(morta[idx]):
            a, b = a + p0, b + p0
            if estat:
                continue
            pr = prov[a:b].copy()
            nz = np.flatnonzero(pr)
            if nz.size == 0:
                out[a:b] = actual if actual else X_
                continue
            # buits: equip anterior; els del principi, el primer equip que surt
            pr[: nz[0]] = pr[nz[0]]
            ffill = np.maximum.accumulate(np.where(pr > 0, np.arange(len(pr)), 0))
            pr = pr[ffill]
            actual = 0  # a la represa sempre hi ha un posseïdor clar
            for eq, ra, rb in trams(pr):
                ra, rb = ra + a, rb + a
                if actual == 0 or mode == "cru":
                    actual = eq
                elif eq != actual:
                    if mode == "acc" and (accions.hi_ha(eq, ra, rb, marge)
                                          or (acc_amb_durada and per_durada(eq, ra, rb))):
                        actual = eq
                    elif mode == "dur" and per_durada(eq, ra, rb):
                        actual = eq
                out[ra:rb] = actual
    return out


def etiqueta(morta, prov, periode, accions, cfg, fps_font, acc_amb_durada=False, control=None):
    k = fps_font / cfg["fps_sortida"]
    marge = int(round(cfg["marge_accio_frames"] * k))
    dmin = int(round(cfg["durada_minima_frames"] * k))
    return {m: aplica_regla(morta, prov, periode, accions, m, marge, dmin, acc_amb_durada, control)
            for m in ("acc", "dur", "cru")}


# ---------------------------------------------------------------- IDSSE

def control_idsse(poss, bola, jxy, equips, c):
    """Frames en què l'equip té BallPossession, un jugador seu a prop de la pilota i la pilota baixa."""
    baixa = np.nan_to_num(bola[:, 2], nan=np.inf) <= c["alcada_max_m"]
    out = {}
    for eq, codi in ((0, H_), (1, A_)):
        d = np.hypot(jxy[:, equips == eq, 0] - bola[:, None, 0], jxy[:, equips == eq, 1] - bola[:, None, 1])
        a_prop = np.nan_to_num(d, nan=np.inf).min(axis=1) <= c["distancia_max_m"]
        out[codi] = (poss == codi) & a_prop & baixa
    return out


def ms(s):
    return int(round(datetime.fromisoformat(s).timestamp() * 1000))


SECCIONS = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}


def idsse_partit(pos_f, cfg):
    arrel = os.path.dirname(pos_f)
    mid = re.search(r"(DFL-MAT-\w+)", pos_f).group(1)
    info = {}
    for _, el in iterparse(glob.glob(os.path.join(arrel, f"DFL_02_01_*{mid}.xml"))[0]):
        if el.tag == "General":
            info.update(el.attrib)
        if el.tag == "Environment":
            info.update({"PitchX": el.get("PitchX"), "PitchY": el.get("PitchY")})
    home, away = info["HomeTeamId"], info["GuestTeamId"]

    pilota, jug = {}, {}
    for ev, el in iterparse(pos_f, events=("start", "end")):
        if ev == "start" and el.tag == "FrameSet":
            sec, equip, pers = el.get("GameSection"), el.get("TeamId"), el.get("PersonId")
            es_pilota = equip.upper() == "BALL"
            buf = []
        elif ev == "end" and el.tag == "Frame":
            g = el.get
            if es_pilota:
                buf.append((int(g("N")), ms(g("T")), float(g("X")), float(g("Y")), float(g("Z") or "nan"),
                            int(g("BallStatus")), int(g("BallPossession") or 0)))
            else:
                buf.append((int(g("N")), float(g("X")), float(g("Y"))))
            el.clear()
        elif ev == "end" and el.tag == "FrameSet":
            if es_pilota:
                pilota[sec] = np.array(buf, dtype=float)
            elif equip in (home, away):
                jug.setdefault(pers, {"equip": equip, "sec": {}})["sec"][sec] = np.array(buf, dtype=float)
            el.clear()

    seccions = sorted(pilota, key=lambda s: SECCIONS.get(s, 9))
    rangs, total = {}, 0
    for s in seccions:
        n0, n1 = int(pilota[s][:, 0].min()), int(pilota[s][:, 0].max())
        rangs[s] = (n0, n1, total)
        total += n1 - n0 + 1
    n = total
    periode = np.zeros(n, np.int8)
    frame_font = np.zeros(n, np.int64)
    t_ms = np.full(n, -1, np.int64)
    bola = np.full((n, 3), np.nan, np.float32)
    estat = np.full(n, -1, np.int8)
    poss = np.zeros(n, np.int8)
    for s in seccions:
        n0, n1, off = rangs[s]
        periode[off:off + n1 - n0 + 1] = SECCIONS.get(s, 9)
        frame_font[off:off + n1 - n0 + 1] = np.arange(n0, n1 + 1)
        b = pilota[s]
        i = b[:, 0].astype(int) - n0 + off
        t_ms[i] = b[:, 1]
        bola[i] = b[:, 2:5]
        estat[i] = b[:, 5]
        poss[i] = b[:, 6]
    persones = sorted(jug, key=lambda p: (jug[p]["equip"] != home, p))
    jxy = np.full((n, len(persones), 2), np.nan, np.float32)
    for j, p in enumerate(persones):
        for s, a in jug[p]["sec"].items():
            if s not in rangs:
                continue
            n0, n1, off = rangs[s]
            m = (a[:, 0] >= n0) & (a[:, 0] <= n1)
            jxy[a[m, 0].astype(int) - n0 + off, j] = a[m, 1:3]

    # events
    events = []
    for _, el in iterparse(glob.glob(os.path.join(arrel, f"DFL_03_02_*{mid}.xml"))[0]):
        if el.tag != "Event":
            continue
        fills = list(el)
        if fills:
            t = ms(el.get("EventTime"))
            x, y = el.get("X-Position"), el.get("Y-Position")
            events.append((t, fills[0], None if x is None else float(x) - 52.5, None if y is None else float(y) - 34.0))
        else:
            el.clear()
    # sincronització: desfasament constant per part, distància mediana Play-pilota mínima
    desf = {}
    for s in seccions:
        n0, n1, off = rangs[s]
        sl = slice(off, off + n1 - n0 + 1)
        tt, bb = t_ms[sl], bola[sl]
        ok = tt >= 0
        tt, bb = tt[ok], bb[ok]
        e = np.array([(t, x, y) for t, f, x, y in events
                      if f.tag == "Play" and x is not None and tt[0] - 10000 <= t <= tt[-1] + 10000])
        millor = None
        for d in range(-5000, 5001, 40):
            i = np.clip(np.searchsorted(tt, e[:, 0] + d), 0, len(tt) - 1)
            dist = float(np.median(np.hypot(bb[i, 0] - e[:, 1], bb[i, 1] - e[:, 2])))
            if millor is None or dist < millor[0] - 1e-9:
                millor = (dist, d)
        desf[s] = {"ms": millor[1], "dist_mediana_m": round(millor[0], 2), "t0": int(tt[0]), "t1": int(tt[-1])}

    def a_frame(t):
        for s in seccions:
            d = desf[s]
            if d["t0"] - 10000 <= t <= d["t1"] + 10000:
                n0, n1, off = rangs[s]
                sl = slice(off, off + n1 - n0 + 1)
                tt = t_ms[sl]
                i = int(np.clip(np.searchsorted(tt, t + d["ms"]), 0, len(tt) - 1))
                return off + i
        return None

    c = cfg["idsse"]["controlades"]
    accions, play_ok, play_n = [], 0, 0
    eq_codi = {home: H_, away: A_}
    for t, f, x, y in events:
        nucli = f
        if f.tag in ("ThrowIn", "FreeKick", "CornerKick", "GoalKick", "KickOff", "Penalty") and list(f):
            nucli = list(f)[0]
        tag = nucli.tag
        if tag == "TacklingGame":
            clau = f"TacklingGame/{nucli.get('WinnerResult')}"
            eq = nucli.get("WinnerTeam")
        elif tag == "BallClaiming":
            clau = f"BallClaiming/{nucli.get('Type')}"
            eq = nucli.get("Team")
        else:
            clau, eq = tag, nucli.get("Team")
        fr = a_frame(t)
        if fr is None or eq not in eq_codi:
            continue
        if clau in c:
            accions.append((fr, fr, eq_codi[eq]))
        if f.tag == "Play" and estat[fr] == 1:
            play_n += 1
            play_ok += int(poss[fr] == eq_codi[eq])

    morta = estat != 1
    valid = (estat >= 0).astype(np.uint8)
    equips = np.array([0 if jug[p]["equip"] == home else 1 for p in persones], np.int8)
    et = etiqueta(morta, poss, periode, Accions(accions), cfg, 25.0,
                  acc_amb_durada=True, control=control_idsse(poss, bola, jxy, equips, cfg["idsse"]["control_per_durada"]))
    return {
        "frame": np.arange(n, dtype=np.int32), "frame_font": frame_font, "period": periode,
        "ball_xyz": bola, "ball_detected": estat >= 0,
        "players_xy": jxy, "players_team": equips,
        "players_id": np.array(persones), "players_detected": np.isfinite(jxy[..., 0]),
        "prov_team": poss, "prov_dead": morta,
        **etiquetes_sortida(et, valid),
        "meta": meta("idsse", mid, home, away, "train", float(info["PitchX"]), float(info["PitchY"]), False, 25.0, cfg,
                      extra={"competicio": f"{info.get('CompetitionName')} {info.get('Season')}",
                             "local_nom": info.get("HomeTeamName"), "visitant_nom": info.get("GuestTeamName"),
                             "sincronitzacio": desf,
                             "comprovacio": {"nom": "Play amb equip = BallPossession", "n": play_n,
                                             "pct": round(100.0 * play_ok / play_n, 2) if play_n else None},
                             "n_accions_controlades": len(accions)}),
    }


# ---------------------------------------------------------------- SkillCorner

def sc_partit(dirp, cfg):
    mid = os.path.basename(dirp)
    with open(os.path.join(dirp, f"{mid}_match.json")) as fh:
        m = json.load(fh)
    home, away = m["home_team"]["id"], m["away_team"]["id"]
    L, W = float(m["pitch_length"]), float(m["pitch_width"])
    jugadors = sorted(((p["team_id"] != home, p["id"]) for p in m["players"]))
    pid = [p for _, p in jugadors]
    slot = {p: i for i, p in enumerate(pid)}

    files = []
    with open(os.path.join(dirp, f"{mid}_tracking_extrapolated.jsonl")) as fh:
        for linia in fh:
            files.append(json.loads(linia))
    files.sort(key=lambda d: d["frame"])
    f0 = files[0]["frame"]
    n = files[-1]["frame"] - f0 + 1
    per = np.zeros(n, np.int8)
    bola = np.full((n, 3), np.nan, np.float32)
    bdet = np.zeros(n, bool)
    grup = np.zeros(n, np.int8)
    jxy = np.full((n, len(pid), 2), np.nan, np.float32)
    jdet = np.zeros((n, len(pid)), bool)
    dades = np.zeros(n, bool)
    for d in files:
        i = d["frame"] - f0
        per[i] = d.get("period") or 0
        b = d.get("ball_data") or {}
        if b.get("x") is not None:
            bola[i] = (b["x"], b["y"], np.nan if b.get("z") is None else b["z"])
        bdet[i] = bool(b.get("is_detected"))
        grup[i] = {"home team": H_, "away team": A_}.get((d.get("possession") or {}).get("group"), 0)
        pdx = d.get("player_data") or []
        dades[i] = len(pdx) > 0 or b.get("x") is not None
        for p in pdx:
            j = slot.get(p["player_id"])
            if j is not None:
                jxy[i, j] = (p["x"], p["y"])
                jdet[i, j] = bool(p.get("is_detected"))

    pp = []
    with open(os.path.join(dirp, f"{mid}_dynamic_events.csv")) as fh:
        for r in csv.DictReader(fh):
            if r["event_type"] != "player_possession":
                continue
            pp.append({"fs": int(r["frame_start"]) - f0, "fe": int(r["frame_end"]) - f0,
                       "eq": H_ if int(r["team_id"]) == home else A_, "end": r["end_type"],
                       "carry": r["carry"] == "True", "gib": r["game_interruption_before"],
                       "gia": r["game_interruption_after"]})
    pp.sort(key=lambda e: (e["fs"], e["fe"]))

    # pilota morta
    c = cfg["skillcorner"]
    fin = int(c["sortida_camp_finestra_s"] * 10)
    morta = np.zeros(n, bool)
    fora = (np.abs(bola[:, 0]) > L / 2) | (np.abs(bola[:, 1]) > W / 2)
    obert, n_fora, n_fallback, n_sense_represa = None, 0, 0, 0

    def inici_morta(e):
        nonlocal n_fora
        w = np.flatnonzero(fora[e["fe"] + 1: e["fe"] + 1 + fin])
        if w.size:
            n_fora += 1
            return e["fe"] + 1 + int(w[0])
        return e["fe"]

    def fi_periode(i):
        q = np.flatnonzero(per == per[i])
        return int(q[-1]) + 1 if per[i] > 0 else i

    # Una parada oberta es tanca a l'inici del player_possession següent: normalment és el que
    # porta game_interruption_before; si no el porta, el comptem.
    anterior = None
    for e in pp:
        if obert is not None:
            if per[obert] != per[e["fs"]]:
                morta[obert:fi_periode(obert)] = True
            elif obert < e["fs"]:
                morta[obert:e["fs"]] = True
            if not e["gib"]:
                n_sense_represa += 1
            obert = None
        elif e["gib"] and anterior is not None and per[anterior["fe"]] == per[e["fs"]]:
            # represa sense parada oberta: la parada comença després de la possessió anterior
            ini = inici_morta(anterior)
            if ini < e["fs"]:
                morta[ini:e["fs"]] = True
            n_fallback += 1
        if e["gia"]:
            obert = inici_morta(e)
        anterior = e
    if obert is not None:
        morta[obert:fi_periode(obert)] = True

    ctrl = c["controlades"]
    accions = [(e["fs"], e["fe"], e["eq"]) for e in pp
               if f"end_type/{e['end']}" in ctrl or (e["carry"] and "carry/True" in ctrl)]
    # comprovació: equip del player_possession = grup majoritari (no nul) durant l'event
    ok = tot = 0
    for e in pp:
        g = grup[e["fs"]:e["fe"] + 1]
        g = g[g > 0]
        if g.size:
            tot += 1
            ok += int(np.bincount(g, minlength=3).argmax() == e["eq"])

    # equip provisional: el del player_possession, mantingut fins al següent
    prov = np.zeros(n, np.int8)
    for e in pp:
        if 0 <= e["fs"] < n:
            prov[e["fs"]] = e["eq"]
    et = etiqueta(morta, prov, per, Accions(accions), cfg, 10.0)
    # possessió crua del proveïdor (possession.group), amb la mateixa pilota morta
    et["prov"] = aplica_regla(morta, grup, per, Accions([]), "cru", 0, 0)
    valid10 = (dades & (per > 0)).astype(np.uint8)

    # 10 -> 25 fps
    n25 = int(np.floor((n - 1) * 2.5)) + 1
    t = np.arange(n25) * 0.4
    nn = np.minimum(np.floor(t + 0.5).astype(int), n - 1)
    i0 = np.minimum(np.floor(t).astype(int), n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (t - i0).astype(np.float32)

    def interp(a):
        a0, a1 = a[i0], a[i1]
        ww = w.reshape((-1,) + (1,) * (a.ndim - 1))
        r = a0 + (a1 - a0) * ww
        exacte = (w == 0).reshape(ww.shape)
        return np.where(exacte, a0, r).astype(np.float32)  # NaN a un extrem -> NaN

    out = {
        "frame": np.arange(n25, dtype=np.int32), "frame_font": (nn + f0).astype(np.int64), "period": per[nn],
        "ball_xyz": interp(bola), "ball_detected": bdet[i0] & np.where(w == 0, True, bdet[i1]),
        "players_xy": interp(jxy), "players_team": np.array([0 if t_ else 1 for t_, _ in jugadors], np.int8),
        "players_id": np.array([str(p) for p in pid]),
        "players_detected": jdet[i0] & np.where((w == 0)[:, None], True, jdet[i1]),
        "prov_team": grup[nn], "prov_dead": morta[nn],
        **etiquetes_sortida({k: v[nn] for k, v in et.items()}, valid10[nn]),
        "meta": meta("skillcorner", mid, home, away, "train", L, W, False, 10.0, cfg, extra={
            "competicio": m["competition_edition"]["name"],
            "local_nom": m["home_team"]["name"], "visitant_nom": m["away_team"]["name"],
            "comprovacio": {"nom": "player_possession amb equip = possession.group majoritari", "n": tot,
                            "pct": round(100.0 * ok / tot, 2) if tot else None},
            "n_parades_amb_sortida_detectada": n_fora, "n_represes_sense_obertura": n_fallback,
            "n_tancades_sense_marca_represa": n_sense_represa, "n_accions_controlades": len(accions)}),
    }
    return out


# ---------------------------------------------------------------- Metrica

def metrica_partit(dirp, cfg):
    nom = os.path.basename(dirp)
    supòsit = True
    L, W = cfg["metrica"]["camp_per_defecte_m"]
    evs = []
    if os.path.exists(os.path.join(dirp, f"{nom}_tracking.txt")):
        from kloppy import metrica
        meta_f = os.path.join(dirp, f"{nom}_metadata.xml")
        ds = metrica.load_tracking_epts(meta_data=meta_f, raw_data=os.path.join(dirp, f"{nom}_tracking.txt"),
                                        coordinates="metrica")
        md = ds.metadata
        L, W, supòsit = float(md.pitch_dimensions.pitch_length), float(md.pitch_dimensions.pitch_width), False
        df = ds.to_df(engine="pandas")
        home_id = [t.team_id for t in md.teams if str(t.ground).lower().endswith("home")][0]
        away_id = [t.team_id for t in md.teams if t.team_id != home_id][0]
        ids, equips = [], []
        for t in md.teams:
            for p in sorted(t.players, key=lambda p: p.player_id):
                if f"{p.player_id}_x" in df:
                    ids.append(p.player_id)
                    equips.append(0 if t.team_id == home_id else 1)
        ordre = sorted(range(len(ids)), key=lambda i: (equips[i], ids[i]))
        ids, equips = [ids[i] for i in ordre], [equips[i] for i in ordre]
        fr = df["frame_id"].to_numpy(int)
        per_raw = df["period_id"].to_numpy(int)
        bxy = df[["ball_x", "ball_y"]].to_numpy(float)
        jxy = np.stack([df[[f"{i}_x", f"{i}_y"]].to_numpy(float) for i in ids], axis=1)
        with open(os.path.join(dirp, f"{nom}_events.json")) as fh:
            for e in json.load(fh)["data"]:
                st = e.get("subtypes")
                st = [st] if isinstance(st, dict) else (st or [])
                evs.append({"eq": H_ if e["team"]["id"] == home_id else A_, "tipus": e["type"]["name"],
                            "sub": "-".join(s["name"] for s in st), "ini": int(e["start"]["frame"]),
                            "fi": int(e["end"]["frame"] or e["start"]["frame"]), "per": int(e["period"])})
    else:
        home_id, away_id = "Home", "Away"
        blocs, ids, equips = {}, [], []
        for k, costat in enumerate(("Home", "Away")):
            with open(os.path.join(dirp, f"{nom}_RawTrackingData_{costat}_Team.csv")) as fh:
                rd = csv.reader(fh)
                next(rd), next(rd)
                cap = next(rd)
                a = np.array([[float(x) if x not in ("", "NaN") else np.nan for x in r] for r in rd])
            blocs[costat] = a
            for i, c in enumerate(cap):
                if c.startswith("Player"):
                    ids.append(f"{costat}_{c}")
                    equips.append(k)
                    blocs.setdefault(costat + "_cols", []).append(i)
            if costat == "Home":
                ib = cap.index("Ball")
        h = blocs["Home"]
        fr = h[:, 1].astype(int)
        per_raw = h[:, 0].astype(int)
        bxy = h[:, [ib, ib + 1]]
        jxy = np.stack([blocs[c][:, [i, i + 1]] for c in ("Home", "Away") for i in blocs[c + "_cols"]], axis=1)
        with open(os.path.join(dirp, f"{nom}_RawEventsData.csv")) as fh:
            for r in csv.DictReader(fh):
                evs.append({"eq": H_ if r["Team"] == "Home" else A_, "tipus": r["Type"], "sub": r["Subtype"] or "",
                            "ini": int(r["Start Frame"]), "fi": int(r["End Frame"] or r["Start Frame"]),
                            "per": int(r["Period"])})
    ordre = np.argsort(fr, kind="stable")
    fr, per_raw, bxy, jxy = fr[ordre], per_raw[ordre], bxy[ordre], jxy[ordre]
    f0 = int(fr[0])
    n = int(fr[-1]) - f0 + 1
    pos = fr - f0
    periode = np.zeros(n, np.int8)
    bola = np.full((n, 3), np.nan, np.float32)
    jug = np.full((n, jxy.shape[1], 2), np.nan, np.float32)
    bola[pos, 0] = (bxy[:, 0] - 0.5) * L
    bola[pos, 1] = (0.5 - bxy[:, 1]) * W
    jug[pos, :, 0] = (jxy[..., 0] - 0.5) * L
    jug[pos, :, 1] = (0.5 - jxy[..., 1]) * W
    for e in evs:
        e["ini"] -= f0
        e["fi"] -= f0
    evs.sort(key=lambda e: (e["ini"], e["fi"]))

    # període vàlid: del primer saque inicial a l'últim event de cada part
    vius = {"PASS", "CARRY", "SHOT", "RECOVERY"}
    n_sense_disparador = 0
    morta = np.zeros(n, bool)
    for p in sorted({e["per"] for e in evs}):
        ep = [e for e in evs if e["per"] == p]
        ko = [e for e in ep if e["tipus"] == "SET PIECE" and "KICK OFF" in e["sub"]]
        a = ko[0]["ini"] if ko else ep[0]["ini"]
        b = max(e["fi"] for e in ep)
        periode[a:b + 1] = p
        for k, e in enumerate(ep):
            if e["tipus"] != "SET PIECE" or e is (ko[0] if ko else None):
                continue
            # disparador més primerenc després de l'últim event viu anterior
            darrer_viu = max((x["ini"] for x in ep[:k] if x["tipus"] in vius and x["ini"] < e["ini"]), default=a)
            disp = []
            for x in ep[:k]:
                if x["tipus"] == "BALL OUT":
                    disp.append(x["fi"])
                elif x["tipus"] == "SHOT" and "GOAL" in x["sub"]:
                    disp.append(x["fi"])  # gol: morta fins al KICK OFF, que és el set piece següent
                elif x["tipus"] == "BALL LOST" and ("OFFSIDE" in x["sub"] or "HAND BALL" in x["sub"]):
                    disp.append(x["fi"])
                elif x["tipus"] == "FAULT RECEIVED" or (x["tipus"] == "CHALLENGE" and "FAULT" in x["sub"]):
                    disp.append(x["ini"])
            disp = [t for t in disp if darrer_viu <= t < e["ini"]]
            if disp:
                morta[min(disp):e["ini"]] = True
            else:
                # sense disparador: morta des del final de l'event anterior
                n_sense_disparador += 1
                prev = max((x["fi"] for x in ep[:k] if x["ini"] < e["ini"]), default=e["ini"])
                if prev < e["ini"]:
                    morta[prev:e["ini"]] = True
        # després de l'últim event, fora del període
    prov = np.zeros(n, np.int8)
    accions = []
    ctrl = set(cfg["metrica"]["controlades"])
    canvia = set(cfg["metrica"]["equip_provisional"])
    for e in evs:
        clau = "CHALLENGE-WON" if e["tipus"] == "CHALLENGE" and "WON" in e["sub"] else e["tipus"]
        if clau in canvia and 0 <= e["ini"] < n:
            prov[e["ini"]] = e["eq"]
        if e["tipus"] in ctrl:
            accions.append((e["ini"], e["fi"], e["eq"]))
    et = etiqueta(morta, prov, periode, Accions(accions), cfg, 25.0)
    valid = (periode > 0).astype(np.uint8)
    return {
        "frame": np.arange(n, dtype=np.int32), "frame_font": np.arange(f0, f0 + n, dtype=np.int64), "period": periode,
        "ball_xyz": bola, "ball_detected": np.isfinite(bola[:, 0]),
        "players_xy": jug, "players_team": np.array(equips, np.int8), "players_id": np.array(ids),
        "players_detected": np.isfinite(jug[..., 0]),
        "prov_team": prov, "prov_dead": morta,
        **etiquetes_sortida(et, valid),
        "meta": meta("metrica", nom, home_id, away_id, "test_only", L, W, supòsit, 25.0, cfg, extra={
            "atribucio": "Metrica Sports sample data, https://github.com/metrica-sports/sample-data",
            "camp_origen": "suposat 105x68 (README de Metrica)" if supòsit else "metadades EPTS",
            "n_set_pieces_sense_disparador": n_sense_disparador, "n_accions_controlades": len(accions)}),
    }


# ---------------------------------------------------------------- comú

def etiquetes_sortida(et, valid):
    # un frame sense etiqueta definida no pot comptar
    valid = valid.astype(bool) & (et["acc"] != X_)
    # label_cru: equip provisional sense regla de duels; label_prov: possessió crua del proveïdor
    return {"label_acc": LLETRES[et["acc"]], "label_dur": LLETRES[et["dur"]], "label_cru": LLETRES[et["cru"]],
            "label_prov": LLETRES[et.get("prov", et["cru"])], "valid": valid.astype(np.uint8)}


def meta(font, partit, home, away, split, L, W, supòsit, fps, cfg, extra):
    return json.dumps({"font": font, "partit": str(partit), "local": str(home), "visitant": str(away),
                       "split": split, "camp_m": [L, W], "camp_suposat": supòsit, "fps_original": fps,
                       "fps": cfg["fps_sortida"], "config_versio": cfg["version"], "config_sha256": cfg["_sha256"],
                       "coordenades": "metres, origen al centre, x llarga, y cap amunt", **extra},
                      ensure_ascii=False, sort_keys=True)


def tasques(src):
    t = []
    fonts = ["idsse", "skillcorner", "metrica"] if src == "all" else src.split(",")
    for f in fonts:
        if f == "idsse":
            t += [("idsse", p) for p in sorted(glob.glob(os.path.join(RAW, "idsse", "DFL_04_03_positions_raw_observed_*.xml")))]
        elif f == "skillcorner":
            t += [("skillcorner", p) for p in sorted(glob.glob(os.path.join(RAW, "skillcorner", "matches", "*")))]
        elif f == "metrica":
            t += [("metrica", p) for p in sorted(glob.glob(os.path.join(RAW, "metrica", "Sample_Game_*")))]
    return t


def fes(args):
    font, path, out, cfg_path = args
    cfg = carrega_config(cfg_path)
    d = {"idsse": idsse_partit, "skillcorner": sc_partit, "metrica": metrica_partit}[font](path, cfg)
    m = json.loads(d["meta"])
    d["meta"] = np.array(d["meta"])
    desti = os.path.join(out, font, f"{m['partit']}.npz")
    desa_npz(desti, d)
    return desti, m.get("comprovacio")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="all", help="all o llista: idsse,skillcorner,metrica")
    ap.add_argument("--out", default="/workspace/shared/bench_v7")
    ap.add_argument("--config", default=os.path.join(AQUI, "label_config.yaml"))
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    feina = [(f, p, a.out, a.config) for f, p in tasques(a.src)]
    with Pool(a.jobs) as pool:
        for desti, comp in pool.imap(fes, feina):
            print(os.path.relpath(desti, a.out), "" if comp is None else f"comprovació {comp['pct']}% (n={comp['n']})", flush=True)


if __name__ == "__main__":
    main()
