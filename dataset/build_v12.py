"""Monta el banco v12 (enlaces, no copias): como v10, pero con el balón muerto visible en movimiento (C11) en IDSSE.
Semillas 0-2: IDSSE v8.2 + C11 (bench_v11), SkillCorner v8.1, Metrica con la degradación de Jan.
Semillas 3-5: IDSSE con degradación de Jan + C11 (bench_v12j), SkillCorner (Jan, equipos corregidos), Metrica con v8.
SkillCorner y Metrica no cambian respecto a v10 (sus features en la VM ya no existen para SkillCorner: la TCN se entrena
en el PC, que las tiene). Las entradas del decodificador y gt_prov solo hacen falta para IDSSE y Metrica.
    python3 build_v12.py"""
import glob, os, re, shutil
H = os.path.expanduser(os.environ.get("MTAH_LOCAL", "~/mtah_data")); SH = os.environ.get("MTAH_SHARED", "/workspace/shared")   # carpeta común de datos del equipo
FONTS = {
    "idsse": [(f"{H}/v11/features", f"{H}/v11/decoder/entrades", f"{H}/bench_v11"), (f"{H}/v12j/features", f"{H}/v12j/decoder/entrades", f"{H}/bench_v12j")],
    "metrica": [(f"{H}/v8/features", f"{H}/v8/decoder/entrades", f"{H}/bench_v8"), (f"{H}/v9/features", f"{H}/v9/decoder/entrades", f"{H}/bench_v9")],
}
DST = [f"{H}/v12/features", f"{H}/v12/decoder/entrades", f"{H}/bench_v12"]
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
for f in glob.glob(f"{H}/v8/decoder/estadistiques_*.json"):
    shutil.copy(f, f"{H}/v12/decoder/")
print(n, "enlaces;", {os.path.basename(d): {f: len(os.listdir(f"{d}/{f}")) for f in FONTS} for d in DST})
