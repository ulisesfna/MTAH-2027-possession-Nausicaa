"""Viterbi amb durades mínimes sobre {H, A, D}.

Cada classe c té una cadena de m_c estats obligatoris i un últim estat amb bucle, de manera que un
tram de c dura com a mínim m_c + 1 frames. Del bucle se surt amb probabilitat p_c cap a una altra
classe c' amb probabilitat T[c, c']. La cadena és determinista, així que no cal desplegar-la: el
millor camí que entra al bucle de c al frame t és el que va entrar a c al frame t - m_c més la suma
de les emissions entremig (sumes acumulades). La recursió és seqüencial en el temps però
vectoritzada sobre un lot de configuracions, totes amb la mateixa longitud.

Emissions log-lineals (vegeu docs/descodificador.md):
    e_t(c) = lam * [ w_t * log(p_tcn_t(c) / prior(c)) + (1 - w_t) * log(q_janv1_t(c)) ]
TCN neutra en forats: quan c.get('r') i r=0 (pilota no visible), p_tcn s'iguala al prior
neutralitzant el terme log(p/prior) = 0 a les emissions, mantenint el pes de janv1.
"""
import numpy as np

CLASSES = ("H", "A", "D")
EPS_Q = 0.1
P_MIN = 1e-6


# ---------------------------------------------------------------- estadístiques d'entrenament

def trams(lab, ok):
    """Trams (classe, inici, fi exclusiu, toca_vora) dins dels segments ok."""
    out = []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], ok.view(np.int8), [0]))))
    for a, b in zip(idx[::2], idx[1::2]):
        x = lab[a:b]
        canvi = np.flatnonzero(x[1:] != x[:-1]) + 1
        ini = np.concatenate(([0], canvi))
        fi = np.concatenate((canvi, [b - a]))
        for i, j in zip(ini, fi):
            out.append((int(x[i]), a + i, a + j, i == 0 or j == b - a))
    return out


def estadistiques(partits):
    """partits: llista de (codis 0/1/2 o -1, període). Torna durades, transicions i prior."""
    durs = {c: [] for c in range(3)}
    trans = np.zeros((3, 3))
    freq = np.zeros(3)
    for y, per in partits:
        ok = (y >= 0) & (per > 0)
        ok[1:] &= per[1:] == per[:-1]  # cap tram no travessa el descans
        freq += np.bincount(y[ok], minlength=3)
        tt = trams(y, ok)
        for k, (c, a, b, vora) in enumerate(tt):
            if not vora:
                durs[c].append(b - a)
            if k + 1 < len(tt) and tt[k + 1][1] == b and tt[k + 1][0] != c:
                trans[c, tt[k + 1][0]] += 1
    return {"p5": [float(np.percentile(durs[c], 5)) for c in range(3)],
            "mitjana": [float(np.mean(durs[c])) for c in range(3)],
            "n_trams": [len(durs[c]) for c in range(3)],
            "transicions": trans.tolist(), "prior": (freq / freq.sum()).tolist()}


def simetritza(st):
    """Estadístiques amb H i A intercanviables: res del partit no diu que el local tingui més pilota
    o possessions més llargues. La TCN es continua dividint pel prior d'entrenament (prior_emissio),
    que és el que va aprendre."""
    out = dict(st)
    mitja = lambda x: [(x[0] + x[1]) / 2, (x[0] + x[1]) / 2, x[2]]
    out["p5"] = mitja(st["p5"])
    out["mitjana"] = mitja(st["mitjana"])
    out["prior_emissio"] = list(st.get("prior_emissio", st["prior"]))
    out["prior"] = mitja(st["prior"])
    T = np.array(st["transicions"], float)
    ha = (T[0, 1] + T[1, 0]) / 2
    viu_d = (T[0, 2] + T[1, 2]) / 2
    d_viu = (T[2, 0] + T[2, 1]) / 2
    out["transicions"] = [[0.0, ha, viu_d], [ha, 0.0, viu_d], [d_viu, d_viu, 0.0]]
    return out


def parametres_hmm(st, s_dur):
    """m_c, log de quedar-se al bucle i log de sortir cap a cada classe.
    s_dur: escalar (aplica a totes les classes) o tupla/llista de 3 multiplicadors (s_H, s_A, s_D)."""
    if isinstance(s_dur, (int, float, np.floating)):
        s = np.full(3, float(s_dur))
    else:
        s = np.asarray(s_dur, float)
    m = np.array([max(1, int(round(s[c] * st["p5"][c]))) for c in range(3)])
    # al bucle s'hi passen (durada - m_c) frames de mitjana: geomètrica amb p = 1 / aquesta mitjana
    p_surt = np.array([1.0 / max(1.0, st["mitjana"][c] - m[c]) for c in range(3)])
    T = np.array(st["transicions"], float)
    T = T / T.sum(1, keepdims=True)
    with np.errstate(divide="ignore"):
        log_surt = np.log(p_surt[:, None] * T)
        log_queda = np.log(1.0 - p_surt)
    np.fill_diagonal(log_surt, -np.inf)
    return m, log_queda, log_surt


# ---------------------------------------------------------------- emissions

def q_janv1(labels):
    """Etiqueta de janv1 suavitzada: 1 - eps a la predita, eps/2 a les altres; X -> uniforme."""
    q = np.full((len(labels), 3), EPS_Q / 2)
    for i, c in enumerate(CLASSES):
        q[labels == c, i] = 1 - EPS_Q
    q[~np.isin(labels, CLASSES)] = 1.0 / 3
    return q


def emissions(p_tcn, q, prior, w, lam):
    """w: escalar o (n,) per frame."""
    w = np.asarray(w, float)
    if w.ndim == 1:
        w = w[:, None]
    lp = np.log(np.maximum(p_tcn, P_MIN)) - np.log(np.asarray(prior))[None]
    return lam * (w * lp + (1 - w) * np.log(q))


# ---------------------------------------------------------------- Viterbi

def viterbi(e, m, log_queda, log_surt, log_inici):
    """e (B, T, 3) emissions; m (B, 3) estats obligatoris; log_queda (B, 3); log_surt (B, 3, 3);
    log_inici (3,). Torna (B, T) amb els codis 0/1/2."""
    B, T, _ = e.shape
    C = np.zeros((B, T + 1, 3))
    np.cumsum(e, axis=1, out=C[:, 1:])
    ent = np.full((B, T, 3), -np.inf)      # millor puntuació en entrar a c al frame t
    arg = np.zeros((B, T, 3), np.int8)      # d'on es venia en entrar
    L = np.full((B, T, 3), -np.inf)        # millor puntuació al bucle de c al frame t
    nou = np.zeros((B, T, 3), bool)         # el bucle s'acaba d'assolir des de la cadena
    bi = np.arange(B)[:, None]
    ci = np.arange(3)[None, :]
    ent[:, 0] = log_inici
    for t in range(T):
        if t > 0:
            cand = L[:, t - 1, :, None] + log_surt          # (B, de, a)
            arg[:, t] = cand.argmax(1)
            ent[:, t] = cand.max(1)
        s = t - m
        val = s >= 0
        sc = np.where(val, s, 0)
        nou_c = np.where(val, ent[bi, sc, ci] + C[:, t][bi, ci] - C[bi, sc, ci], -np.inf)
        queda = L[:, t - 1] + log_queda if t > 0 else np.full((B, 3), -np.inf)
        nou[:, t] = nou_c > queda
        L[:, t] = np.maximum(nou_c, queda) + e[:, t]
    out = np.empty((B, T), np.int8)
    for b in range(B):
        entrades = [np.flatnonzero(nou[b, :, c]) for c in range(3)]
        t = T - 1
        c = int(L[b, t].argmax())
        while t >= 0:
            k = np.searchsorted(entrades[c], t, side="right") - 1
            if k < 0 or not np.isfinite(L[b, t, c]):
                out[b, :t + 1] = c  # no hauria de passar si el tram és més llarg que m
                break
            s = entrades[c][k] - m[b, c]
            out[b, s:t + 1] = c
            if s <= 0:
                break
            c, t = int(arg[b, s, c]), s - 1
    return out


def descodifica(p_tcn, labels_janv1, dubtos, period, st, configs, conf=None, r=None):
    """configs: llista de dicts {w, lam, s_dur, porta, [factor], [r]}. Torna (len(configs), n) amb H/A/D/X.

    Es descodifica cada part per separat; fora de les parts, X."""
    n = len(labels_janv1)
    # Lots de 384 configs: amb meitats de ~75k frames el pic de RAM queda en ~3 GB (viterbi +
    # emissions + còpia d'entrada), dins dels ~9 GB lliures; reduïs un 33% les iteracions de bucle
    # per frame respecte 192 (2 blocs en comptes de 3 per a 480 configs).
    if len(configs) > 384:
        out = np.empty((len(configs), n), dtype="<U1")
        for i in range(0, len(configs), 384):
            out[i:i + 384] = descodifica(p_tcn, labels_janv1, dubtos, period, st, configs[i:i + 384], conf=conf, r=r)
        return out
    q = q_janv1(np.asarray(labels_janv1).astype("<U1"))
    prior = np.array(st["prior"])
    # prior amb què es divideix la TCN: el d'entrenament, encara que el de l'HMM sigui simètric
    prior_e = np.array(st.get("prior_emissio", st["prior"]))
    out = np.full((len(configs), n), "X", dtype="<U1")
    _clau_s = lambda s: tuple(s) if isinstance(s, (list, tuple, np.ndarray)) else s
    # clau d'ordenació tolerant a barrejar escalars i tuples (els tuples van darrere)
    _ord_s = lambda k: (isinstance(k, tuple), k if isinstance(k, tuple) else (k,))
    claus_s = sorted({_clau_s(c["s_dur"]) for c in configs}, key=_ord_s)
    hmm = {k: parametres_hmm(st, k) for k in claus_s}
    for h in (1, 2):
        idx = np.flatnonzero(period == h)
        if idx.size == 0:
            continue
        a, b = idx[0], idx[-1] + 1
        es, ms, lq, ls = [], [], [], []
        for c in configs:
            p = np.where(r[a:b, None], p_tcn[a:b], prior_e[None]) if (c.get("r") and r is not None) else p_tcn[a:b]
            w = c["w"]
            if "factor" in c and conf is not None:
                w = w * (1.0 - c["factor"] * conf[a:b])
            if c.get("porta"):
                w = np.where(dubtos[a:b], w, 0.0)
            es.append(emissions(p, q[a:b], prior_e, w, c["lam"]))
            m, log_queda, log_surt = hmm[_clau_s(c["s_dur"])]
            ms.append(m)
            lq.append(log_queda)
            ls.append(log_surt)
        # allibera la llista de 384 emissions per-config abans d'empilar (estalvia ~1,4 GB al pic)
        es_stack, es = np.stack(es), None
        codis = viterbi(es_stack, np.stack(ms), np.stack(lq), np.stack(ls), np.log(prior))
        del es_stack
        out[:, a:b] = np.array(CLASSES)[codis]
    return out
