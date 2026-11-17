# REPORT_DATASET — corrección del banco externo (bench v8)

Iniciado el 16-nov-2026. Carpeta de código: `modelos_ulises/dataset_v8/`. Datos generados: `~/mtah_data/bench_v8/`
(propio, no compartido; nada se escribe en `/workspace/shared`).

## Por qué

El modelo `ulises` mejora +0.057 de spellF1 en el banco de Jan (degradación v3) y nada en los 5 min públicos; toda la
familia con TCN queda ahí por debajo de las reglas solas. Un clasificador separa banco y BAR-ATH con AUC 0.99
(`ulises/REPORT.md`, iteraciones 3-4). Hipótesis: el banco degradado no se parece lo bastante al tracking del
concurso y los modelos aprenden atajos que solo existen en el banco.

## Qué está permitido y qué no (corrige una suposición de partida)

- **Sí se puede inspeccionar el tracking y el vídeo del BAR-ATH** para entenderlos (RULES §1: "Watching the video and
  inspecting the data to understand them, debug your pipeline and design features"). Jan ya calibra la degradación con
  estadísticas agregadas del tracking del partido, sin etiquetas (`docs/degradacio.md`, "excepció documentada").
- **No se puede usar la GT pública de 5 min como conjunto de ajuste** (RULES §2: "not training data and not a tuning
  set"; issue #16). Ajustar el dataset hasta que las notas de los modelos en el banco coincidan con sus notas en los
  5 min sería ajustar con esas etiquetas de forma indirecta: los parámetros del dataset, y los modelos entrenados con
  él, quedarían derivados de la GT pública. **Esa idea, tal cual, no la implemento.**
- Lo que sí hago, que consigue lo mismo sin etiquetas del partido:
  1. **Parecido del tracking** (entradas): estadísticas agregadas del balón, jugadores y huecos, y AUC de un
     clasificador banco-vs-partido. El objetivo de cada iteración es bajar esa distancia.
  2. **Parecido del comportamiento de los modelos, sin etiquetas**: se pasan varios modelos (reglas, baseline, TCN,
     fusio_v7d, ulises) por el banco y por el partido y se comparan cosas que no necesitan GT: % de balón muerto que
     predicen, posesiones por minuto, y el acuerdo entre modelos. Si el banco se parece al partido, un mismo modelo
     debe comportarse igual en los dos. Es la versión legítima de la idea de "probar todos los modelos".
  3. **Los 5 min públicos se miran una vez al final**, como informe, no como objetivo a optimizar.

## Diagnóstico 1 — tracking bruto, BAR-ATH frente a banco v3 (`explora.py`)

Solo entradas, sin etiquetas del partido. IDSSE = 2 partidos de ejemplo, semilla 0.

| | BAR-ATH | IDSSE v3 | Metrica | SkillCorner |
|---|---|---|---|---|
| balón visible | 0.593 | 0.55-0.61 | 0.61-0.63 | 0.64 |
| **balón parado (< 0.05 m/s) entre los frames visibles** | **0.5 %** | **12-16 %** | 0.1-0.6 % | 0.1-0.3 % |
| aceleración del balón (2ª diferencia), p50 / p95 | 5.7 / 23 | **14-18 / 160-230** | 0.8 / 13-20 | 2.8 / 47 |
| balón fuera del campo (frames visibles) | 4.2 % | 9.5 % | 1.4-3.5 % | 6-8 % |
| duración de los tramos visibles, p25 / p50 | 5.5 / 13.8 s | **1.4 / 5.2-5.6 s** | 6.5-8 / 15-17 s | 10 / 26 s |
| **al empezar un hueco: velocidad del balón (p25/p50/p75)** | 0.6 / 1.6 / 4.0 | **0 / 0 / 0** | 5.9 / 9.9 / 16 | 2.5 / 4.6 / 7.3 |
| al empezar un hueco: distancia a la línea, p50 | 6.0 m dentro | **0.2-0.5 m fuera** | 4-5.6 m | 0.7 m |
| al empezar un hueco: jugador más cercano, p50 | 1.45 m | 3.0-3.8 m | 2.4-2.9 m | 2.8-3.1 m |
| **al acabar un hueco: velocidad del balón (p25/p50/p75)** | **5.3 / 11.4 / 17.1** | **0 / 0 / 3.5-4.7** | 0.4 / 3.6 / 8.1 | 2.3 / 4.7 / 7.8 |
| al acabar un hueco: jugador más cercano, p50 | 0.68 m | 3.9-4.7 m | 0.00 m | 1.9-2.3 m |
| desplazamiento del balón durante el hueco, p50 | 10.2 m | **0 m** | 3.7-5.6 m | 8.6-14.8 m |

Lectura:
1. **En el BAR-ATH el balón desaparece en movimiento, cerca de un jugador y dentro del campo, y reaparece ya
   golpeado (11 m/s) junto a un jugador, 10 m más allá.** En IDSSE v3 pasa lo contrario: el balón se "apaga" cuando
   está parado fuera del campo y se "enciende" en el mismo sitio, todavía parado. La v3 forzó que el 90 % de los
   frames de hueco cayeran en balón muerto (referencia SkillCorner) multiplicando la tasa por la etiqueta, y eso
   llevó los huecos al balón parado.
2. **En el BAR-ATH casi no existe balón visible y parado** (0,5 %); en IDSSE v3 es el 12-16 %. "Balón visible y
   muerto" significa cosas distintas en el banco y en el concurso. Encaja con lo visto en `ulises`: los modelos con
   TCN detectan en el partido 5-9 puntos menos de balón muerto que las reglas.
3. **El balón de IDSSE es mucho más ruidoso** (aceleraciones 3-10 veces mayores); el del ATD está muy suavizado
   (Jan ya lo midió: residuo de 0,6 mm). La degradación actual no suaviza.
4. IDSSE v3 tiene muchos tramos visibles muy cortos (p25 1,4 s frente a 5,5 s).
5. Metrica tiene un balón sintético (pegado al jugador, trayectorias a trozos lineales). Es de solo test y del mismo
   proveedor que la GT; no se toca en esta primera versión.

## Plan de correcciones (v8, solo IDSSE en la primera versión)

| id | corrección | de dónde sale el parámetro |
|---|---|---|
| C1 | Suavizar balón y jugadores (Savitzky-Golay) | ventana que iguala los cuantiles de aceleración a los del BAR-ATH |
| C2 | Ocultar el balón parado | umbral y duración mínima para igualar la fracción de balón lento visible del BAR-ATH |
| C3 | Huecos que empiezan según velocidad × jugador cercano × línea (modelo multiplicativo) **sin usar la etiqueta** | tasas de inicio medidas en el BAR-ATH por celda |
| C4 | Los huecos acaban en un golpeo (balón rápido junto a un jugador) | cuantiles de velocidad al reaparecer del BAR-ATH |
| C5 | Mantener de Jan: relleno de huecos de 0,64-5,12 s, fragmentación de tracks, slots anónimos | `target_stats.json` |

## Herramientas de validación (todas sin etiquetas del BAR-ATH)

| herramienta | qué mide | uso |
|---|---|---|
| `sig.descriptors` + `calib8.distancia` | ~20 agregados del tracking (velocidad y aceleración del balón, balón lento, fuera del campo, duración de tramos visibles y huecos, contexto al empezar y acabar un hueco). Distancia = error relativo simétrico medio frente al BAR-ATH (0 = idéntico) | **objetivo de la calibración** |
| `probes.py` + `compara_probes.py` | Comportamiento de 4 modelos fijos (reglas janv1, baseline, TCN v7d, fusio_v7d) en el partido y en el banco: % D que predicen, posesiones/hora, D con balón visible, vivo sin balón, acuerdo entre modelos. z = distancia del partido a la distribución de los 7 IDSSE | **validación** (las sondas de reglas y baseline entran también en la calibración desde la fase 3; las de TCN/fusio nunca) |
| `adv_ds.py` | AUC de un clasificador banco-vs-partido sobre las 35 features de la TCN con contexto ±1 s (entrena con 4 IDSSE + 1ª parte, evalúa con 3 IDSSE + 2ª parte) | **validación independiente** |

## Registro de cambios

### v8a — primera versión (16-nov, 11:45 UTC). `degrade8.py`, `target8.py`, `calib8.py`

Mecanismo (solo IDSSE; no usa ninguna etiqueta, a diferencia de la v3):
- **C1** Savitzky-Golay del balón (ventana 55 frames) y de los jugadores (15).
- **C2** el balón parado (< 0.1 m/s durante ≥ 3 s) se oculta.
- **C3** huecos adicionales con la tasa de inicio del BAR-ATH por celda (modelo multiplicativo velocidad × jugador
  cercano × línea; en el partido la tasa es 3.4× con el balón a < 1 m/s y 2.4× a < 2 m de la línea, y no depende del
  jugador más cercano), duración muestreada de la distribución del partido.
- **C4** el hueco acaba en un golpeo (jugador a < 1.2 m y balón a ≥ 7.5 m/s en los 5 frames siguientes) con
  probabilidad 0.7; si no, cuando el balón se mueve.
- **C5** de Jan: relleno de huecos de 0.64-5.12 s, fragmentación de tracks, slots anónimos.

Calibración: búsqueda por coordenadas sobre 16 parámetros minimizando la distancia de descriptores en 3 IDSSE
(semilla 0); comprobada en los 7. Tres fases (`calib8_fase12.log`, `calib8.log`).

| descriptores, 7 IDSSE semilla 0 | BAR-ATH | v3 (Jan) | v8a |
|---|---|---|---|
| **distancia global** | 0 | **0.424** | **0.149** |
| balón visible | 0.593 | 0.593 | 0.493 ✗ |
| balón parado (< 0.05 m/s) / visible | 0.5 % | 13.2 % | 0.6 % |
| aceleración del balón > 30 m/s² | 1.8 % | 16.6 % | 1.3 % |
| aceleración del balón p50 / p90 | 5.7 / 18.3 | 13.5 / 45.5 | 6.2 / 17.0 |
| golpeo junto a jugador (acc > 15, d1 < 1 m) | 11.4 % | 47.6 % | 7.5 % |
| huecos largos por hora | 74.8 | 109.2 | 75.7 |
| tramos visibles p25 / p50 | 5.5 / 13.8 s | 1.4 / 5.7 s | 7.2 / 14.6 s |
| huecos p50 | 10.7 s | 9.5 s | 16.4 s ✗ |
| velocidad al empezar el hueco p50 / p75 | 1.5 / 3.9 | 0 / 0 | 0.4 / 2.0 |
| velocidad al reaparecer p25 / p50 / p75 | 5.0 / 12.2 / 17.2 | 0 / 0 / 5.1 | 4.7 / 7.0 / 13.2 |
| jugador más cercano al reaparecer p50 | 0.67 m | 4.7 m | 1.1 m |
| desplazamiento durante el hueco p50 | 11.5 m | 0 m | 11.2 m |

Validación por comportamiento de modelos (`compara_probes.py v3 v8`; z del partido frente a los 7 IDSSE):

| estadístico | BAR-ATH | v3 (z) | v8a (z) |
|---|---|---|---|
| acuerdo reglas-TCN | 0.791 | 0.868 (−3.6) | 0.837 (**−1.8**) |
| acuerdo reglas-TCN con balón visible | 0.737 | 0.828 (−5.1) | 0.779 (**−1.1**) |
| % D de la TCN menos % D de las reglas | −0.050 | −0.008 (−7.0) | −0.024 (**−1.6**) |
| vivo sin balón, TCN | 0.060 | 0.030 (2.4) | 0.061 (**0.0**) |
| vivo sin balón, baseline | 0.084 | 0.123 (−4.2) | 0.085 (**−0.2**) |
| posesiones/h de la TCN sin decodificar | 382 | 521 (−1.3) | 354 (0.5) |
| **D con balón visible, reglas** | 0.088 | 0.087 (0.1) | 0.037 (**11.0 ✗**) |
| **D con balón visible, TCN** | 0.062 | 0.081 (−2.2) | 0.025 (**8.9 ✗**) |
| balón visible | 0.593 | 0.593 (0.0) | 0.493 (2.3 ✗) |
| posesiones/h de las reglas | 195 | 170 (1.4) | 151 (4.2 ✗) |
| \|z\| medio / mediano / nº con \|z\| > 2 | | 1.98 / 1.30 / 13 | 2.07 / 1.15 / 11 |

Lectura: la v8a arregla lo que motivó el trabajo (los modelos con TCN ya se desvían de las reglas en el banco casi
como en el partido, y el balón tiene la cinemática correcta) pero **rompe otra cosa**: oculta demasiado balón
durante los parones (D con balón visible 3.7 % frente a 8.8 %). En conjunto empata con la v3, con errores distintos.

Notas de los modelos en IDSSE (label_acc, semilla 0) — la v8a es un banco más difícil: reglas 0.560 → 0.498,
fusio_v7d 0.710 → 0.632, TCN sin decodificar 0.313 → 0.360.

### Por qué la v8a oculta demasiado, y corrección (v8c)

- El balón de IDSSE está parado ≥ 3 s el ~40 % del tiempo (parones). Con los huecos de C3 encima, el balón visible
  no pasa del 49 % con ningún valor de los parámetros (fases 3 y 4 de la calibración, variantes con k = 0.2 y 0).
- Descomposición del "balón visible cuando las reglas dicen D" (`explora3.py`): en el BAR-ATH es el 8.8 % del tiempo
  = fuera del campo 2.3 + **dentro y lento (< 1 m/s) 2.1** + libre a > 3 m de todos 2.2 + resto 2.1. En la v8a el
  término "dentro y lento" es 0.2. En el partido hay un 5.4 % del tiempo de balón visible y lento; en la v8a, 2.1 %.
- Es decir: **en el BAR-ATH sí se ve balón parado, pero con un vaivén del tracker de 0.1-1 m/s**, no a velocidad
  exactamente 0. Mi descriptor "< 0.05 m/s" me llevó a ocultar todo el balón parado (C2), y eso era un error.
- Corrección: **C2b** solo se oculta una fracción `p_parat` de los parones; **C7** se añade al balón un vaivén lento
  (AR(1) en posición, σ `sigma_w`). Los dos parámetros entran en la calibración.

### v8c — parones parcialmente visibles y vaivén del balón (12:15 UTC)

Calibración (fase 5, objetivo = descriptores + sondas de reglas/baseline con peso 4): elige `p_parat` = 0.6 (se oculta
el 60 % de los parones), `sigma_w` = 0.10 m, `v_parat` = 0.3, golpeo a ≥ 5 m/s.

| 3 IDSSE, semilla 0 | BAR-ATH | v8a | v8c |
|---|---|---|---|
| balón lento visible: < 0.3 m/s / < 1 m/s (sobre visible) | 3.3 % / 9.9 % | 2.2 % / 6.8 % | 4.3 % / 10.9 % |
| D con balón visible, reglas | 0.088 | 0.036 | 0.063 |
| balón visible | 0.593 | 0.49 | 0.51 |
| distancia de descriptores | 0 | 0.143 | 0.151 |

**Decisión: aceptada.** Recupera el balón muerto visible sin perder la cinemática. El balón visible total sigue 7-8
puntos por debajo del partido (ver limitaciones).

### v8 (final) — jugadores: caídas ligadas al balón y equipo erróneo (12:45 UTC)

Lo que separaba el banco del partido tras arreglar el balón eran los jugadores (`explora4.py`, `explora5.py`,
`target8_jug.py`). En el BAR-ATH:
- Con el balón visible se ven 21.4 jugadores de media (los 22 el 61 % del tiempo); **con el balón oculto, 19.2**.
  El 4.4 % del tiempo hay menos de 14 jugadores (el 95 % de esas veces sin balón): la cámara deja de ver el campo.
  En el banco de Jan la visibilidad de los jugadores no depende del balón y nunca baja de 17.
- **El 12.9 % del tiempo un equipo tiene 12 detecciones** (74 veces por hora, mediana 2.9 s), y el 59 % de esas veces
  el total sigue siendo 22: es un jugador asignado al equipo contrario. En el banco eso no ocurre nunca.

Correcciones:
- **C8** caídas individuales con tasa distinta según se vea o no el balón, y caídas masivas (10-100 % de los
  jugadores) dentro de los huecos largos de balón, con la duración de las rachas del partido.
- **C9** con la tasa del partido, una detección pasa al equipo contrario (85 %) o se duplica (15 %).
Ajuste a mano en 4 pasadas (`cal_jug.py`) contra los agregados del partido:

| 7 IDSSE, semilla 0 | BAR-ATH | v3 (Jan) | v8 |
|---|---|---|---|
| jugadores visibles con balón / sin balón | 21.4 / 19.2 | 20.5 / 20.4 | 21.6 / 19.3 |
| tiempo con < 8 / < 14 / < 18 / < 20 jugadores | 2.1 / 4.4 / 9.2 / 14.9 % | 0 / 0 / 1.4 / 18.5 % | 3.0 / 4.1 / 5.8 / 11.7 % |
| tiempo con un equipo de 12 | 12.9 % | 0 % | 15.0 % |
| cambios del nº de visibles por minuto | 19.8 | 23.2 | 24.0 |

## v8.1 — SkillCorner re-degradado (16-nov, 14:30 UTC)

**Por qué.** SkillCorner son 20 de los 27 partidos con que entrena la TCN y seguía con la degradación de Jan.
Medido con `desc_fonts.py skillcorner` y `adv_ds.py` (`ADV_FONT=skillcorner`), sin etiquetas:

| | BAR-ATH | SkillCorner sin degradar | v3 (Jan) |
|---|---|---|---|
| jugadores visibles por frame | 20,5 | 9,5 | 9,5 |
| balón visible | 0,593 | 0,598 | 0,638 |
| aceleración del balón > 30 m/s² | 1,9 % | 10,5 % | 10,0 % |
| huecos largos por hora / mediana | 75 / 10,7 s | 112 / 5,7 s | 57 / 15,9 s |
| velocidad del balón al acabar un hueco (mediana) | 12,2 m/s | 5,1 | 4,8 |
| **AUC banco-vs-partido** (0,5 = indistinguible) | — | — | **0,998** |

El clasificador separa SkillCorner v3 del partido casi sin error, y lo hace por el número de jugadores visibles:
Jan descarta las posiciones extrapoladas del proveedor y se queda con las ~9,5 detectadas por la cámara de
televisión; el ATD del partido ve ~20,5. La TCN ha aprendido el 74 % de sus partidos con medio campo vacío.

**Variantes probadas** (7 partidos, semilla 0; parámetros de IDSSE v8 salvo lo indicado):

| variante | qué cambia | AUC | decisión |
|---|---|---|---|
| v3 (Jan) | — | 0,998 | referencia |
| A | suavizado C1 + parones C2 + vaivén C7, sin huecos nuevos, solo jugadores detectados | 0,996 | descartada |
| B | A + huecos C3 (`k=0.3`) y huecos propios alargados hasta un golpeo (`cop_natural`) | 0,995 | descartada |
| D | A + jugadores extrapolados del proveedor como vistos (`extrap=1`) + caídas C8/C9 | 0,883 | descartada (C es mejor) |
| **C** | B + D | **0,856** | **aceptada** |

**Qué hace la v8.1 en SkillCorner** (`degrade8.py genera --font skillcorner --params params_v8.json --set k=0.3 extrap=1 cop_natural=1`):
- `extrap=1`: las posiciones extrapoladas de SkillCorner cuentan como jugador visto (15,6 visibles por frame; no
  llega a 20,5 porque el proveedor tampoco da posición para todos). Encima se aplican las caídas C8 y los errores
  de equipo C9, igual que en IDSSE.
- Los huecos de balón propios de SkillCorner se conservan (son de un tracker de televisión real). Se añade: C1
  (suavizado), C2 (parte de los parones se oculta), C3 con `k=0.3` (pocos huecos nuevos: ya tiene los suyos),
  `cop_natural=1` (el hueco propio largo también se alarga hasta el siguiente golpeo), C5 (relleno 0,64–5,12 s), C7.
- Resultado en descriptores (4 partidos): balón visible 0,564 (objetivo 0,593), huecos largos 70/h (75),
  aceleración > 30 m/s²: 0,6 % (1,9 %), tramo visible mediano 18 s (13,8 s).

**Lo que no arregla.** La velocidad al reaparecer sigue baja (4,0 m/s frente a 12,2): con el balón de SkillCorner
(10 fps, interpolado) y el suavizado de 2,2 s casi ningún frame cumple la condición de golpeo. Los jugadores
extrapolados son posiciones inventadas por el proveedor: cerca del balón casi siempre hay detección real, pero
lejos pueden estar mal. El AUC 0,856 sigue lejos del 0,744 de IDSSE.

**Metrica no se toca**: nunca entra en entrenamiento y es el único conjunto que ningún modelo ha visto; se deja
igual que en v3 para que sus notas sean comparables con todo lo anterior.

## v8.2 — fallo encontrado: equipos de SkillCorner invertidos (16-nov, 18:15 UTC)

**Primer reentreno con v8/v8.1 (TCN entrenada en el PC de Ulises, resto en la VM): resultado malo, y por qué.**

| | banco v3, modelos v3 | banco v8, modelos reentrenados con v8.1 |
|---|---|---|
| fusio (spellF1, 10 partidos) | 0,696 | 0,556 |
| `ulises` | 0,753 | 0,676 |
| `ulises`, solo Metrica (mismo fichero en los dos bancos) | 0,710 | 0,676 |
| 5 min públicos (informativo): fusio / `ulises` | 0,477 / 0,333 | 0,438 / 0,474 |
| BAR-ATH sin etiquetas: % D de la TCN − % D de las reglas | −0,050 | −0,017 |
| BAR-ATH sin etiquetas: acuerdo por frame TCN–reglas | 0,791 | **0,547** |

La última fila delata el problema: la TCN nueva confunde local y visitante. En el banco, fuera de pliegue, acierta
el 51–73 % de los frames de IDSSE (la v7d, 89–93 %) y manda a "visitante" entre el 50 y el 76 % de los frames de
posesión local.

**Causa.** En los 20 partidos de SkillCorner del banco del equipo (`/workspace/shared/bench_v7`, y por tanto en
todas las degradaciones v1–v3 y en mi v8.1) `players_team` está invertido respecto a las etiquetas H/A. Medido con
las reglas janv1, que solo miran el tracking: aciertan el 19–40 % de los frames tal cual y el 75–87 % si se
intercambian los equipos; en IDSSE y Metrica aciertan el 83–89 % tal cual. El origen es `bench/build_labels.py`,
línea 461 (`0 if t_ else 1`).

**Por qué no se había notado.** Con la degradación de Jan, SkillCorner tiene 9,5 jugadores visibles y se distingue de
IDSSE trivialmente, así que la TCN v7d aprendió dos convenios, uno por fuente (acierta ~90 % en ambas). Al hacer que
SkillCorner se parezca a IDSSE (v8.1), ya no puede saber qué convenio toca y mezcla los equipos. Dicho de otro modo:
**la TCN v7d del equipo solo aprende local/visitante de forma transferible con los 7 IDSSE**; los 20 SkillCorner le
enseñan el convenio contrario.

**Corrección (`fix_equips_sc.py`).** En mi copia (`~/mtah_data/bench_v8/skillcorner`) se invierte `players_team` en
los 60 ficheros (queda marcado en `deg_meta`: `equips_corregits`, versión v8.2). Comprobación: las reglas pasan de
~0,25 a 0,785 y 0,841 en dos partidos. No se toca nada de la zona común; hay que avisar a Jan.
Features y entradas de SkillCorner regeneradas; resultado del reentreno con v8.2 en la sección siguiente.
Los resultados del reentreno fallido quedan en `ulises/results_v8_sc_invertido/`.

## Resultado del reentreno con v8.2 (16-nov, 20:50 UTC)

TCN (8 pliegues + final) y stack entrenados en el PC de Ulises (x86, 8 hilos); el resto en la VM.
Los bancos son distintos entre columnas: solo Metrica y las filas "BAR-ATH sin etiquetas" se comparan directamente.

| | modelos de Jan/`ulises` entrenados con el banco de Jan (deg. v3) | reentrenados con v8.1 (equipos SC invertidos) | reentrenados con **v8.2** |
|---|---|---|---|
| TCN fuera de pliegue, exactitud por frame en IDSSE | 0,89–0,93 | 0,51–0,73 | 0,88–0,93 |
| fusio, spellF1 en su banco (10 partidos) | 0,696 | 0,556 | 0,694 |
| `ulises`, spellF1 en su banco | 0,753 | 0,676 | 0,722 [0,681, 0,756] |
| `ulises` − fusio (pareado) | +0,057 [+0,042, +0,074], 10/10 | +0,120 | +0,028 [+0,017, +0,037], 9/10 |
| Metrica (mismo fichero en los tres): fusio / `ulises` | 0,676 / 0,710 | 0,609 / 0,676 | 0,663 / 0,679 |
| BAR-ATH sin etiquetas: % D de la TCN / de las reglas | 0,409 / 0,458 | 0,442 / 0,458 | **0,464 / 0,458** |
| BAR-ATH sin etiquetas: D con balón visible, TCN / reglas | 0,062 / 0,088 | 0,091 / 0,088 | 0,101 / 0,088 |
| BAR-ATH sin etiquetas: acuerdo por frame TCN–reglas | 0,791 | 0,547 | **0,813** |
| BAR-ATH: posesiones y % D de fusio / de `ulises` | 208, 41,6 % / 227, 37,9 % | — | 156, 45,8 % / 197, 40,5 % |
| 5 min públicos (informativo, 17 posesiones): fusio / `ulises` | 0,477 / 0,333 | 0,438 / 0,474 | 0,477 / 0,492 |

**Lectura.**
1. La corrección de equipos funciona: la TCN vuelve a acertar ~90 % en IDSSE y ya no mezcla local y visitante.
2. **El síntoma que motivó todo este trabajo desaparece en la TCN**: en el partido predice tanto balón muerto como
   las reglas (0,464 frente a 0,458; antes 0,409) y coincide más con ellas por frame (0,813 frente a 0,791). Es una
   medida sin etiquetas: dice que la TCN se comporta en el partido como en el banco, no que acierte más.
3. **En Metrica, que es el mismo fichero en todos los casos, el reentreno no mejora**: fusio 0,676 → 0,663 y
   `ulises` 0,710 → 0,679. Son 3 partidos; la diferencia está dentro del ruido, pero el signo es negativo.
4. La ganancia del stack sobre fusio se reduce a la mitad (+0,028) aunque sigue siendo significativa (9 de 10).
5. El stack vuelve a bajar el balón muerto en el partido (40,5 % frente al 45,8 % de fusio): la selección anidada
   elige ahora el bono a D igual a 0.
6. fusio reentrenada da solo 156 posesiones en el partido (antes 208): fuera del intervalo que el equipo considera
   normal para el partido; `ulises` da 197.
7. En los 5 min públicos todo queda entre 0,48 y 0,49 (v9m1 del equipo: 0,483; reglas solas: 0,571). Con 17
   posesiones no distingue nada, y no se usa para decidir.

**Conclusión.** El banco v8.2 es un banco mejor (más parecido al partido en las tres validaciones y sin el fallo de
equipos), pero **no hay evidencia de que un modelo reentrenado con él puntúe más en el partido**: la señal sin
etiquetas es favorable, Metrica es ligeramente desfavorable y la ventana pública no discrimina. Modelo empaquetado
en `ulises/model_v8/`, CSV `ulises/out/ulises_v8.csv` (sha256 d1d9e28d…8f61), reproducido byte a byte por
`predict.py` en la VM.

## v11 — C11: balón muerto visible en movimiento (17-nov, mañana)

**Por qué.** En el BAR-ATH, donde reglas y TCN dicen balón muerto con el balón visible, el balón se mueve (mediana
1,41 m/s; 24 % de los frames por debajo de 0,3 m/s). En el banco v8.2 está casi quieto (0,38 m/s; 43 %) y con la
degradación de Jan totalmente quieto (0,00 m/s; 82 %). El stack de `ulises` aprende de esa diferencia y en el partido
descarta balón muerto que TCN y reglas sí ven (E18/E19 en `ulises/REPORT.md`).

**Mecanismo** (`carry11.py`, post-proceso sobre ficheros ya degradados, sin etiquetas): tramos visibles con velocidad
suavizada (1 s) < 0,8 m/s durante ≥ 75 frames; con probabilidad 0,8, hasta 4 excursiones de ida y vuelta (seno; empiezan
y acaban en el punto original, así que no hay saltos), cada una de ≤ 250 frames, con velocidad media lognormal
(mediana 2,0 m/s, σ log 0,5); el último segundo y medio (o un tercio del tramo si es corto) queda quieto.

**Calibración** (3 IDSSE, semilla 0; frames D visibles según la etiqueta externa, frente al objetivo del partido sin etiquetas):

| | q10 / q25 / mediana / q75 / q90 (m/s) | fracción < 0,3 m/s |
|---|---|---|
| BAR-ATH (reglas D y TCN D, balón visible) | 0,10 / 0,31 / 1,41 / 4,16 / 11,1 | 0,244 |
| v8.2 antes | 0,12 / 0,23 / 0,42 / 1,60 / 5,59 | 0,357 |
| **v8.2 + C11 (v11)** | 0,19 / 0,38 / 1,25 / 3,15 / 6,28 | 0,187 |
| degradación de Jan antes / con C11 | 0 / 0 / 0 / 0 / 0 → 0 / 0 / 0 / 1,7 / 3,4 | 0,909 → 0,62 |

En la degradación de Jan apenas sirve (su balón muerto visible son trozos cortos y exactamente quietos), así que el
banco v11 es solo IDSSE v8.2 + C11 (semillas 0-2), con Metrica igual que en v8.2. Datos en `~/mtah_data/bench_v11/idsse`.

## v9 — intento de corregir balón visible y duración de huecos (16-nov, 15:15 UTC). NO adoptada

Objetivo: las dos diferencias que la v8 deja en IDSSE (balón visible 0,51–0,52 frente a 0,593; hueco mediano 17 s
frente a 10,7 s). Todo sobre 3 IDSSE, semilla 0, mientras la TCN se reentrena con la v8.

**Barrido 1 (`sweep9_a.log`): un parámetro cada vez.** `escala_dur=0.25`, `p_parat=0.4`, `busca_cop=25`, `k=0.7`,
`p_cop=0.4`, `max_enrere=100`. Ninguno baja la mediana de los huecos de 16–18 s; `escala_dur=0.25` sube el balón
visible a 0,54 y nada más. Conclusión: la duración no la fijan los huecos de C3 sino los de C2, que ocultan el
parón entero hasta el siguiente golpeo, y los parones de la Bundesliga duran lo que duran.

**Idea C10 (`dur_parat`).** El hueco de un parón dura como un hueco largo del partido (muestra de
`durada_llarg_q101` × `dur_parat`) y después el balón reaparece parado, con su vaivén.

**Barrido 2 (`sweep9_b.log`) y puntuación (`eval9.py` → `eval9.json`)**, con la misma distancia sin etiquetas de la
calibración (descriptores del tracking + sondas de reglas y baseline; menor es mejor):

| candidato | balón visible | hueco mediano | dist. descriptores | dist. descr.+sondas | % D reglas | posesiones/h reglas |
|---|---|---|---|---|---|---|
| BAR-ATH (objetivo) | 0,593 | 10,7 s | 0 | 0 | 0,458 | 195 |
| **v8** | 0,509 | 17,1 s | **0,152** | **0,117** | 0,518 | 157 |
| a: `dur_parat=1.0 p_parat=0.85` | 0,519 | 11,8 s | 0,199 | 0,140 | 0,526 | 163 |
| b: `dur_parat=1.5 p_parat=0.85` | 0,518 | 13,0 s | 0,188 | 0,136 | 0,515 | 164 |
| c: `dur_parat=1.5 p_parat=1.0 k=0.8` | 0,524 | 11,7 s | 0,200 | 0,138 | 0,524 | 166 |
| d: a + `escala_dur=0.3` | 0,558 | 11,2 s | 0,208 | 0,141 | 0,503 | 178 |

**Decisión: ningún candidato se adopta; la v8 sigue siendo el banco.** C10 arregla lo que buscaba (el candidato d
deja el balón visible en 0,558, el hueco mediano en 11,2 s y acerca las sondas: % D de las reglas 0,503, 178
posesiones/h, % D del baseline 0,346 frente a 0,323), pero la distancia global empeora en todos (0,117 → 0,136–0,141)
porque rompe dos cosas que la v8 había acertado: el balón reaparece parado (velocidad mediana al reaparecer 3,2 m/s
frente a 12,2 en el partido; la v8 da 5,7) y hay el doble o el triple de balón parado visible (6–9 % frente a 3,3 %).
Es un compromiso real, no un fallo de calibración: en IDSSE el parón dura más que el hueco del partido, así que o
el hueco es demasiado largo o el balón se ve parado demasiado tiempo. Resolverlo exigiría otro mecanismo (por
ejemplo, que el balón reaparezca en movimiento antes del saque: el recogepelotas o el jugador llevándolo), que no
está implementado. El código de C10 queda en `degrade8.py` desactivado (`dur_parat=0`).

## Resultado: v3 de Jan frente a v8 (7 IDSSE; todo sin etiquetas del BAR-ATH)

| validación | v3 (Jan) | v8a (solo balón) | **v8** |
|---|---|---|---|
| Distancia de descriptores del tracking (0 = igual al partido) | 0.424 | 0.149 | **0.154** |
| **AUC adversarial** sobre las features de la TCN (0.5 = indistinguible), validación independiente | 0.900 | 0.844 | **0.744** |
| Comportamiento de modelos: \|z\| medio (33 estadísticos) | 1.98 | 2.07 | **1.49** |
| … estadísticos con \|z\| > 2 | 13 | 11 | **10** |
| % D de la TCN menos % D de las reglas (partido −0.050) | −0.008 (z −7.0) | −0.024 (−1.6) | −0.026 (**−1.2**) |
| D con balón visible según la TCN (partido 0.062) | 0.081 (z −2.2) | 0.025 (8.9) | 0.047 (**1.1**) |
| vivo sin balón según la TCN (partido 0.060) | 0.030 (z 2.4) | 0.061 (0.0) | 0.053 (**0.3**) |
| acuerdo reglas-TCN con balón visible (partido 0.737) | 0.828 (z −5.1) | 0.779 (−1.1) | 0.802 (−2.9) |
| balón visible (partido 0.593) | 0.593 (z 0.0) | 0.493 (2.3) | 0.523 (2.1) |
| posesiones/h de las reglas (partido 195) | 170 (z 1.4) | 151 (4.2) | 156 (5.2) |

Tras la v8 lo que más separa banco y partido son la dispersión y el centro de los equipos y la zona del campo donde
está el balón: es el estilo de juego de ese partido concreto, no el tracker, y no se debe "corregir".

### Las notas de los modelos cambian (el banco v8 es más difícil)

label_acc, 7 IDSSE, semilla 0, modelos SIN reentrenar (TCN v7d entrenada con la v3):

| modelo | banco v3 | banco v8 | BAR-ATH, 5 min públicos (informativo) |
|---|---|---|---|
| reglas janv1 `p3_notag` | 0.560 | 0.495 | 0.533 |
| baseline_smooth50 | 0.294 | 0.318 | – |
| TCN v7d sin decodificar | 0.313 | 0.403 | 0.218 |
| fusio_v7d | 0.710 | 0.657 | 0.477 |

Única mirada a los 5 min, solo como informe: en la ventana pública las reglas quedan por encima de fusio_v7d
(0.533 frente a 0.477). En el banco fusio_v7d gana a las reglas por +0.150 con la v3 y por +0.162 con la v8: **el
banco v8 sigue sin reproducir ese orden**. Con 17 posesiones la ventana no distingue 0.48 de 0.53, así que no es una
contradicción, pero tampoco una confirmación de que el banco ya sea fiel. No se ha ajustado nada con estos números.

## Limitaciones conocidas de la v8

1. **Balón visible 52 % frente a 59 %** y huecos más largos (mediana 17 s frente a 10.7 s). El balón de IDSSE está
   parado en los parones ~40 % del tiempo; o los parones de la Bundesliga son más largos que los del BAR-ATH, o el
   tracker del concurso muestra más balón en los parones de lo que el mecanismo permite. No se ha forzado.
2. **Las reglas hacen menos posesiones por hora en el banco** (156 frente a 195): señal de que al banco le falta
   algo que en el partido parte las posesiones (¿huecos cortos?, ¿más errores de equipo cerca del balón?).
3. El mecanismo es una hipótesis razonada ("el tracker pierde el balón lento u ocluido y lo recupera en un golpeo;
   la cámara a veces deja de ver el campo"), calibrada con agregados; no sabemos cómo funciona el tracker real.
4. **Solo se ha rehecho IDSSE.** SkillCorner (20 partidos de entrenamiento de la TCN) y Metrica (3 de test) siguen
   siendo los de Jan (enlaces). Metrica tiene un balón sintético pegado al jugador; no se ha tocado.
5. La calibración usa agregados del tracking del BAR-ATH y el comportamiento de reglas/baseline sobre él (sin
   ninguna etiqueta). Es la misma excepción que documenta Jan en `docs/degradacio.md`, algo más amplia: hay que
   contarla en el informe técnico.
6. El suavizado del balón (ventana de 55 frames, 2.2 s) lo elige la calibración por la distribución de
   aceleraciones; deforma los pases cortos. Es lo que dicen los datos del ATD, pero es agresivo.
7. Las etiquetas (`label_acc`, etc.) no cambian: siguen siendo las de Jan. La duda de qué convención se parece a la
   GT del concurso sigue abierta y este trabajo no la toca.

## Qué falta para saber si sirve (siguiente fase, no hecha)

El dataset corregido solo vale si un modelo **reentrenado** con él se comporta mejor. Pasos, con coste en esta VM:
1. Features de `bench/features.py` sobre `~/mtah_data/bench_v8` (minutos).
2. Reentrenar la TCN con validación cruzada (8 pliegues × ~21 min ≈ 3 h) con `bench/train_tcn.py` apuntando a v8.
3. Entradas del decodificador, fusio y `ulises` sobre v8 (≈ 1 h).
4. Comprobar sin etiquetas que la TCN nueva ya no ve menos balón muerto que las reglas en el BAR-ATH
   (`probes.py`), y solo entonces mirar una vez la ventana pública.
Riesgo: SkillCorner sigue con la degradación antigua y es el 74 % de los partidos de entrenamiento de la TCN.

## Cómo reproducir

```
cd modelos_ulises/dataset_v8 ; P=/workspace/shared/venvs/bench/bin/python ; V=../../.venv/bin/python
$P target8.py ; $P target8_jug.py          # agregados del BAR-ATH (sin etiquetas) -> target8*.json
$P probes.py partit ; $P probes.py v3      # comportamiento de los modelos sonda
$P calib8.py ; $P calib8.py amplia ; PES_SONDES=4 $P calib8.py sondes     # calibración (fases 1-5)
$P degrade8.py genera --seeds 0 1 2 --set $(python -c "import json;print(' '.join(f'{k}={v}' for k,v in json.load(open('params_v8.json')).items()))")
$P probes.py v8 ; $P compara_probes.py v3 v8 ; $V adv_ds.py v3 v8          # validación
```
Parámetros finales: `params_v8.json`. Salida: `~/mtah_data/bench_v8/{idsse,skillcorner,metrica}` (≈1,1 GB; IDSSE y, desde la v8.1, SkillCorner
son propios; Metrica son enlaces a la v3 de Jan). Mismo formato npz que `bench_v7d_deg`, así que el código de Jan lo lee sin cambios.
