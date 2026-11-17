# Aviso para el equipo: equipos invertidos en SkillCorner (16-nov-2026)

**Qué pasa.** En los 20 partidos de SkillCorner de `/workspace/shared/bench_v7` (y por tanto en `bench_v7_deg`,
`bench_v7b_deg`, `bench_v7d_deg`, las features y todo lo que se entrena con ellos) `players_team` está invertido
respecto a las etiquetas `label_*`: el equipo 0 del tracking es el que las etiquetas llaman A.

**Cómo se ve** (sin entrenar nada; reglas janv1 `p3_notag` frente a `label_acc`, exactitud por frame):

| fuente | tal cual | con los equipos intercambiados |
|---|---|---|
| SkillCorner (20) | 0,19–0,40 | 0,75–0,87 |
| IDSSE (7) | 0,83–0,89 | 0,43–0,56 |
| Metrica (3) | 0,83 | 0,32–0,39 |

Con `label_dur` sale lo mismo. Se reproduce con las entradas ya cacheadas en
`/workspace/shared/decoder_v7d/entrades/skillcorner/*_s0.npz` (campos `janv1` y `gt_acc`).

**Dónde está.** `bench/build_labels.py`, línea 461:
`"players_team": np.array([0 if t_ else 1 for t_, _ in jugadors], np.int8)`. En IDSSE (línea 305) es
`0 if equip == home else 1`. No he tocado ese fichero ni nada de la zona común.

**Por qué no se había notado.** La TCN acierta ~90 % en las dos fuentes: SkillCorner degradado tiene 9,5 jugadores
visibles por frame y se distingue de IDSSE sin esfuerzo, así que aprende un convenio de equipos por fuente. Al
hacer que SkillCorner se parezca a IDSSE (jugadores extrapolados, dataset v8.1), la TCN reentrenada mezcla local y
visitante (51–73 % de exactitud en IDSSE fuera de pliegue).

**Qué implica.** La TCN v7d aprende local/visitante de forma transferible solo con los 7 IDSSE; los 20 SkillCorner
le enseñan el convenio contrario. Hipótesis sin medir: corregirlo y reentrenar podría mejorar la TCN del equipo
también en el banco v3.

**Corrección mínima.** Invertir `players_team` en SkillCorner (o intercambiar H/A en sus etiquetas) y regenerar
degradación, features y TCN. En mi copia está hecho con `dataset_v8/fix_equips_sc.py` (solo `~/mtah_data/bench_v8`).
