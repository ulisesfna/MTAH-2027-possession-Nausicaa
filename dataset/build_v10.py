"""Monta el banco v10 = las dos degradaciones a la vez, con enlaces (no copia datos).
Semillas 0-2: degradación propia v8.2 (IDSSE, SkillCorner) y Metrica con la degradación de Jan (= banco v8.2 tal cual).
Semillas 3-5: degradación de Jan (IDSSE; SkillCorner con equipos corregidos) y Metrica con la degradación v8.
    python3 build_v10.py"""
import glob, os, re, shutil
H = os.path.expanduser(os.environ.get("MTAH_LOCAL", "~/mtah_data")); SH = os.environ.get("MTAH_SHARED", "/workspace/shared")   # carpeta común de datos del equipo
FONTS = {  # (features, entradas del decodificador, banco) para las semillas 0-2 y 3-5
    "idsse": [(f"{H}/v8/features", f"{H}/v8/decoder/entrades", f"{H}/bench_v8"), (f"{SH}/features_v7d", f"{SH}/decoder_v7d/entrades", f"{SH}/bench_v7d_deg")],
    "skillcorner": [(f"{H}/v8/features", f"{H}/v8/decoder/entrades", f"{H}/bench_v8"), (f"{H}/v10j/features", f"{H}/v10j/decoder/entrades", f"{SH}/skillcorner_corregit/bench_v7_deg")],
    "metrica": [(f"{H}/v8/features", f"{H}/v8/decoder/entrades", f"{H}/bench_v8"), (f"{H}/v9/features", f"{H}/v9/decoder/entrades", f"{H}/bench_v9")],
}
DST = [f"{H}/v10/features", f"{H}/v10/decoder/entrades", f"{H}/bench_v10"]
n = 0
for font, grups in FONTS.items():
    for k, origens in enumerate(grups):
        for src, dst in zip(origens, DST):
            os.makedirs(f"{dst}/{font}", exist_ok=True)
            for f in sorted(glob.glob(f"{src}/{font}/*_s[012].npz")):
                p, ll = re.match(r"(.*)_s(\d)\.npz$", os.path.basename(f)).groups()
                d = f"{dst}/{font}/{p}_s{int(ll) + 3 * k}.npz"
                if os.path.lexists(d):
                    os.remove(d)
                os.symlink(os.path.realpath(f), d); n += 1
for f in glob.glob(f"{H}/v8/decoder/estadistiques_*.json"):   # mismas etiquetas -> mismas estadísticas del HMM
    shutil.copy(f, f"{H}/v10/decoder/")
print(n, "enlaces;", {os.path.basename(d): {f: len(os.listdir(f"{d}/{f}")) for f in FONTS} for d in DST})
