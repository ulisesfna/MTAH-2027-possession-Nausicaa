# ulises — registro de mejora sobre fusio_v7d

Iniciado el 16-nov-2026. Trabajo autónomo de Claude en `modelos_ulises/ulises/`. Formato "Guess. Test. Keep or kill"
(lección de RawTree): cada idea con su cifra y su decisión.

## Protocolo (fijado antes de mirar resultados)

- **Base:** fusio_v7d de Jan (rama `origin/v9-markius`, commit `9d278a5`, exportada sin tocar a `base/`):
  TCN no causal v7d + reglas janv1 `p3_notag` + Viterbi con duraciones mínimas.
- **Banco:** el de Jan, `MTAH_DEG=v3`, 10 partidos de selección (7 IDSSE + 3 Metrica), semillas 0-2, probabilidades de
  la TCN fuera de pliegue. Todo se lee de `/workspace/shared` (no se escribe nada allí).
- **Selección anidada idéntica a la de Jan:** para cada IDSSE que se deja fuera, los hiperparámetros se eligen con los
  otros 6 IDSSE (`label_acc`); para Metrica, con los 7 IDSSE. Ningún número de la tabla es dentro de muestra.
- **Regla de aceptación:** Δ spellF1 pareado por partido con IC95 (bootstrap entre partidos, 2000, semilla 0).
  Se acepta si el IC de `label_acc` no cruza 0 **y** no empeora de forma significativa con `label_dur` ni con
  `label_prov` (canal del proveedor, el más parecido a la GT del concurso en IDSSE). Si el IC cruza 0: "no se
  distinguen, gana el más simple".
- **Nada usa el BAR-ATH** ni la ventana pública, ni para ajustar ni para elegir.
- Laboratorio: `lab.py` (protocolo + métrica rápida), `cvit.c` (Viterbi en C, verificado idéntico al de Jan y ~200×
  más rápido). Resultados en bruto: `results/*.json`.

## Punto de partida reproducido

| modelo | spellF1 acc [IC95] | dur | prov | macroF1 | IDSSE acc | Metrica acc | nº spells pred/GT |
|---|---|---|---|---|---|---|---|
| fusio_v7d (`base_fusio`) | 0.6963 [0.664, 0.728] | 0.6921 | 0.4914 | 0.8907 | 0.7052 | 0.6755 | 205 / 242 |

Coincide con el banco de Jan del 13-nov (0.696 [0.664, 0.728], dur 0.692).

## Diagnóstico (antes de probar nada)

`diag1.py`, `diag2.py`, predicciones fuera de pliegue de fusio_v7d:

1. **Se pierden las posesiones cortas.** Recall a IoU 0.5 por duración de la posesión GT (IDSSE / Metrica):
   < 2 s: 0.13 / 0.09 · 2-4 s: 0.40 / 0.36 · 4-10 s: 0.64 / 0.66 · > 10 s: 0.84-0.87 / 0.77-0.83.
   Las de < 4 s son el 21 % (IDSSE) y el 27 % (Metrica) de las posesiones y cada una cuenta igual que una larga.
2. **Causa:** la mayoría de las cortas perdidas no se predicen en absoluto ("nada": el modelo sigue con el rival) o
   quedan fusionadas en una posesión mucho más larga. El modelo predice 205 posesiones por partido frente a 242.
3. **La información existe en parte:** en ~40 % de las cortas perdidas, la TCN o las reglas aciertan en más de la mitad
   de sus frames, pero el decodificador las alisa. El argmax de la TCN sin decodificar da 3,35× posesiones (spellF1
   0.32); las reglas solas, 1,13× (0.56).
4. **Bordes:** las transiciones con D están bien fechadas (mediana 0 f, p10-p90 ±17 f). Las H↔A tienen el 45 % (IDSSE)
   y el 32-36 % (Metrica) a más de 1 s, sin sesgo sistemático (mediana −2…+3 f): no hay un desfase global que corregir.

Conclusión: el margen está en (a) evidencia por frame que no borre las posesiones cortas y (b) un decodificador que
no las alise. Coincide con `docs/lecciones_rawtree.md` (el margen está con balón visible).

## Registro de experimentos

| id | hipótesis | Δ acc [IC95] (n/10) | Δ dur | Δ prov | decisión |
|---|---|---|---|---|---|
| E1 | La rejilla de Jan está en un borde: λ más alto (hasta 0.5) y duración mínima 0.25·p5 | +0.0015 [−0.0036, +0.0054] (7) | −0.0058 [−0.0146, +0.0018] | +0.0092 | **Descartada**: no se distingue en acc y tiende a empeorar en dur |
| E2 | M1 de Marc: duración mínima distinta para D | +0.0010 [−0.0001, +0.0022] (6) | +0.0007 | +0.0009 | **Descartada**: efecto ≈ 0 (gana el más simple) |
| E3 | Coste de transición ajustable (más o menos cambios) | +0.0005 [−0.0038, +0.0034] (9) | −0.0051 | +0.0070 | **Descartada**: ≈ 0 |
| E4 | Pesos libres TCN / reglas y confianza de las reglas (ε) en vez de w·λ con ε = 0.1 | **+0.0073 [+0.0040, +0.0107] (9)** | +0.0005 [−0.0083, +0.0076] | +0.0089 [+0.0064, +0.0116] | Mejora real sobre fusio_v7d (elige a = 0.25, b = 0.05, ε = 0.02 en 6/8 pliegues), pero **queda fuera del modelo final**: no se suma al stack (E8b) |
| E5 | **Modelo apilado**: un LightGBM por encima de la TCN (probabilidades fuera de pliegue en ±2 s y medias móviles, reglas, features locales), entrenado solo con IDSSE, sustituye a la TCN en el decodificador | **+0.0417 [+0.0248, +0.0599] (10)** | **+0.0324 [+0.0166, +0.0489]** | **+0.0294 [+0.0187, +0.0403]** | **Aceptada**: 0.7380 [0.701, 0.770]. IDSSE +0.053, Metrica +0.015 (los 3 partidos y las 3 etiquetas mejoran). El decodificador elige b = 0 (las reglas ya están dentro del stack) |

### Notas de E5 (`stack.py`, `exp_stack.py`)
- Entrenamiento: probabilidades de la TCN v7d **fuera de pliegue** de 6 IDSSE × 3 semillas (1 de cada 4 frames),
  aumentación intercambiando equipos, promedio simétrico al predecir. Un modelo por IDSSE dejado fuera; Metrica se
  predice con el modelo de los 7 IDSSE. SkillCorner no entra (sus probabilidades TCN no son fuera de pliegue).
- Simplificación declarada: al elegir los parámetros del decodificador para el pliegue de p, las probabilidades de
  los partidos de ajuste q vienen del stack que deja fuera q (y ha visto p como entrenamiento). El partido evaluado p
  nunca lo ha visto ni su TCN ni su stack. Pendiente: comprobar con doble anidado.
- Lo que más pesa en el stack: medias móviles de 1 s y 3 s de las probabilidades de la TCN, la aceleración del balón
  y el tiempo desde/hasta el cambio de las reglas. (Corregido tras la ablación E11: el alisado aprendido solo explica +0.008; el resto viene de
  las reglas y las features locales.)
- Riesgo: el stack solo ve IDSSE degradado. Metrica (mismo proveedor que la GT, nunca en entrenamiento) mejora menos
  que IDSSE; es la señal de transferencia que hay que vigilar en cada iteración.

## Iteración 2 — variantes del stack

| id | hipótesis | vs fusio_v7d: Δ acc [IC95] | vs E5 (`a0`): Δ acc / dur / prov | decisión |
|---|---|---|---|---|
| E6 (`a1`) | Ponderar cada frame de H/A por 1/duración de su posesión (la métrica cuenta posesiones, no frames) | +0.0450 [+0.0286, +0.0609] | +0.0033 [−0.0047, +0.0120] / −0.0090 [−0.023, +0.007] / **+0.0237 [+0.018, +0.029]** | **Descartada por ahora**: no se distingue en acc, tiende a empeorar en dur; solo gana claro en prov. Gana el más simple (sin ponderar) |
| E7 (`t0`) | **Secuencia de contactos** (idea touchseq de El Chato v5): por equipo, tiempo desde el último / hasta el siguiente contacto (jugador a < 1 m y < 2 m del balón) y "golpeo" (contacto con aceleración del balón > 15 m/s²), y qué equipo tocó antes / tocará después. Solo usa columnas de `features.py` | **+0.0639 [+0.0436, +0.0854] (10/10)** | **+0.0222 [+0.0120, +0.0329]** / +0.0015 [−0.009, +0.011] / **+0.0253 [+0.018, +0.033]** | **Aceptada**: 0.7602 [0.722, 0.788]; dur 0.7260, prov 0.5461. Metrica +0.034 sobre fusio_v7d (antes +0.015) |

Incidencia: `t0` y la ablación `tcn0` estuvieron varias horas sin arrancar por un error mío en el script de espera
(se detectaba a sí mismo). Relanzadas a las 06:45. Cada variante tarda 25-50 min en esta VM (8 modelos).

## Iteración 3 — chequeo de dominio y papel de las reglas en el decodificador

Motivación: la prueba en el partido (sección siguiente) y un chequeo **sin etiquetas del partido** (`diag4.py`):

| | balón visible | % D según reglas | % D según TCN (argmax) | D con balón visible: reglas / TCN |
|---|---|---|---|---|
| IDSSE (7, banco) | 0.55-0.62 | 42-54 % | 40-53 % (± 1 punto de las reglas) | 0.066-0.096 / 0.066-0.093 (iguales) |
| Metrica (3, banco) | 0.61-0.69 | 30-41 % | 32-42 % | 0.015-0.056 / 0.015-0.054 (iguales) |
| **BAR-ATH** | 0.593 | **45,8 %** (`p3_notag`) · **50,0 %** (`p3` + Smart Tagging) | **40,9 %** | 0.088 (0.129 con ST) / **0.062** |

En el banco, reglas y TCN coinciden en cuánto balón muerto hay; en el BAR-ATH la TCN ve 5-9 puntos menos de D que las
reglas, y el stack todavía menos (35-37 %). Es una señal de **desplazamiento de dominio en el balón muerto** (no dice
quién acierta, pero sí que la TCN se comporta distinto que en el banco). Por eso probé dar a las reglas más papel en
el decodificador, validándolo solo en el banco:

| id | hipótesis | vs `t0` (b = 0): Δ acc [IC95] / dur / prov | decisión |
|---|---|---|---|
| E8 | Peso fijo b de las reglas en las emisiones (todas las clases): b = 0.05 / 0.10 / 0.20 | −0.0057 [−0.012, +0.001] / −0.0191 [−0.029, −0.009] / −0.0491 [−0.067, −0.029] | **Descartada**: b = 0.05 no se distingue; b ≥ 0.10 empeora de forma significativa |
| E8b | Igual con reglas "seguras" (ε = 0.02, lo que ganaba en E4): b = 0.05 / 0.10 | −0.0121 [−0.021, −0.003] / −0.0363 | **Descartada**. De paso: la mejora de E4 no se suma al stack (el stack ya usa las reglas) |
| E9 | **Bono solo a D donde las reglas dicen D** (d): 0.1 / 0.2 / 0.4 / 0.8 / 1.6 | +0.0008 / +0.0016 [−0.001, +0.005] / +0.0026 [−0.001, +0.007] (prov **+0.0079 [+0.005, +0.012]**, Metrica +0.007) / −0.0018 / −0.0105 [−0.018, −0.003] | **Aceptada como opción neutra**: óptimo interior en d ≈ 0.2-0.4; nunca empeora y mejora algo prov y Metrica. Metida en la rejilla anidada (`g3`, d ∈ {0, 0.2, 0.4}) |

Estado tras la iteración 3 — **ulises = TCN v7d + stack `t0` + decodificador `g3`**:
0.7607 [0.723, 0.788] (Δ +0.0644 [+0.0447, +0.0848], 10/10) · dur 0.7258 (+0.0338) · prov 0.5465 (+0.0551) ·
macroF1 0.9131 · IDSSE 0.7812 · Metrica 0.7129. Parámetros del pliegue `tots` (los que se usarían en el partido):
a = 0.25, b = 0, d = 0.2, s_dur = 0.5.

## Iteración 4 — validación adversarial (banco vs BAR-ATH, sin etiquetas) y robustez de las features

Idea de `docs/lecciones_rawtree.md` (DreamTeam: validación adversarial por feature). `adv.py`: un clasificador
banco-vs-partido sobre las features del stack y el AUC univariante de cada una. Solo entradas del partido, ninguna
etiqueta. (Zona gris declarada: se usa el tracking del BAR-ATH sin etiquetas para decidir qué features son fiables,
igual que Jan lo usó para calibrar la degradación v3.)

- El clasificador separa banco y partido casi perfectamente (AUC 0.986 con IDSSE, 0.993 con Metrica): el banco
  degradado no es indistinguible del concurso.
- **Las features más desplazadas son las de "golpeo" que añadí en E7** (contacto con aceleración del balón > 15 m/s²):
  |AUC − 0.5| = 0.12-0.19. Tasas medidas con balón visible:

| | jugador a < 1 m | aceleración > 15 m/s² | **las dos a la vez (golpeo)** | p90 de la aceleración |
|---|---|---|---|---|
| IDSSE (7) | 0.31-0.36 | 0.14-0.21 | 0.08-0.13 | 22-32 |
| Metrica (3) | 0.42-0.53 | 0.14-0.16 | 0.10-0.12 | 30-33 |
| **BAR-ATH** | 0.29 | 0.16 | **0.03** | **18** |

  En el concurso el balón casi nunca acelera fuerte cuando está junto a un jugador (3 % frente a 8-13 %): la
  trayectoria del balón del ATD es distinta cerca de los jugadores (más suave o desfasada). Los contactos por
  distancia sí son comparables (0.29 frente a 0.31-0.36).
- Consecuencia: parte de la mejora de E7 (+0.022) se apoya en una señal que en el partido no existe igual.

| id | hipótesis | resultado | decisión |
|---|---|---|---|
| E10 (`t2`) | Quitar las features de golpeo y dejar solo los contactos por distancia: se pierde poco en el banco y se gana robustez | 0.7533 [0.717, 0.783]; frente a `t0`: acc −0.0074 [−0.014, −0.001], **dur +0.0048 [+0.001, +0.009]**, prov −0.0058 [−0.013, +0.002]. Metrica igual (0.7103 frente a 0.7129 en acc; mejor en dur 0.665 frente a 0.654 y en prov 0.630 frente a 0.619) | **Aceptada (decisión de robustez)**: ninguna de las dos domina en todas las etiquetas; los golpeos solo ayudan en IDSSE, no en Metrica, y son la señal más desplazada respecto al BAR-ATH. Me quedo con `t2` aunque cueste 0.007 en `label_acc` |
| E11 (`tcn0`) | Ablación: stack solo con las probabilidades de la TCN (sin reglas, sin features locales, sin contactos) | 0.7046: solo +0.0083 [+0.0020, +0.0154] sobre fusio_v7d, y −0.0334 [−0.047, −0.020] frente a E5 | **Conclusión**: la ganancia del stack no viene de alisar mejor la TCN sino de la información añadida (reglas + features locales con contexto temporal + contactos). Corrige mi nota de E5 |

Comprobación de la simplificación del anidado (pendiente de E5): con **una sola configuración fija para todos los
pliegues** (a = 0.35, s_dur = 0.5, d = 0.2, sin ninguna selección) `t2` da 0.7543 / dur 0.7316 / prov 0.5413, igual que
con selección anidada (0.7533). El resultado no depende de cómo se eligen los parámetros del decodificador, así que el
doble anidado no hace falta.

Rediagnóstico con `t2` (`diag1.py stack_t2_g3 t2`): recall a IoU 0.5 por duración, antes → ahora (IDSSE):
1-2 s 0.13 → 0.22 · 2-4 s 0.40 → 0.52 · 4-10 s 0.64 → 0.73 · 10-20 s 0.84 → 0.87 · > 20 s 0.87 → 0.93.
Metrica: 0.09 → 0.17 · 0.36 → 0.42 · 0.66 → 0.73 · 0.83 → 0.86 · 0.77 → 0.79. Las posesiones cortas siguen siendo
el mayor agujero (se predicen 214 de 242).

**Estado tras la iteración 4 — ulises = TCN v7d + stack `t2` + decodificador `g3`:**
0.7533 [0.717, 0.783] (Δ +0.0570 [+0.0421, +0.0741], 10/10) · dur 0.7306 (+0.0386 [+0.024, +0.053], 10/10) ·
prov 0.5406 (+0.0493 [+0.038, +0.061], 10/10) · macroF1 0.9114 · IDSSE 0.7718 · Metrica 0.7103 (+0.035).

## Iteración 5 — qué se le pide al stack que aprenda

| id | hipótesis | vs `t2`: Δ acc [IC95] / dur / prov | decisión |
|---|---|---|---|
| E12 (`t3`) | Ponderación intermedia por posesión (peso ∝ duración^−0.5) con las features de `t2`: recupera posesiones cortas sin el coste en dur que tenía E6 | −0.0007 [−0.006, +0.005] / −0.0065 [−0.015, +0.003] / **+0.0139 [+0.011, +0.017]** (10/10). Predice 221 posesiones en vez de 214. Metrica: acc −0.004, dur +0.012, prov +0.011; IDSSE: dur −0.014 | **Descartada con la regla fijada** (acc no se distingue → gana el más simple). Nota: es la mejor variante en `label_prov` (0.5545); si el equipo decide priorizar la etiqueta del proveedor, es la candidata |
| E13 (`p2`) | Entrenar el stack con `label_prov` (la etiqueta derivada del tracking del proveedor, la más parecida a cómo se hizo la GT del concurso) en vez de `label_acc` | **−0.0179 [−0.034, −0.002]** / **−0.0325 [−0.059, −0.003]** / **+0.0496 [+0.041, +0.057]** (10/10). Por fuente: IDSSE acc −0.024, dur −0.061, prov +0.057; **Metrica acc −0.004, dur +0.034, prov +0.033** | **Descartada con la regla fijada** (empeora de forma significativa en acc y dur). Pero deja una pregunta abierta para el equipo: en Metrica (mismo proveedor que la GT) entrenar con la etiqueta del proveedor no empeora acc y mejora las otras dos. El modelo aprende la convención con la que se le entrena; cuál se parece más a la GT del concurso no se puede decidir con el banco |

## Prueba en el partido objetivo (ventana pública de 5 min) — SOLO INFORMATIVO

`match_entry.py` + `match_predict.py`. Mismo código que el banco (adapter → features → TCN v7d final → reglas
`p3_notag` → stack de los 7 IDSSE → decodificador con los parámetros del pliegue `tots`). Comprobación del pipeline:
mi fusio_v7d da exactamente lo que el banco de Jan del 5-nov (208 posesiones, 41,59 % D, público 0.4774 / 0.6742).

| modelo | público spellF1 | público macroF1 | posesiones pred/GT (ventana) | partido: posesiones / dur. media / % D |
|---|---|---|---|---|
| reglas janv1 `p3` con Smart Tagging (banco de Jan) | 0.571 | 0.855 | – | 301 / – / 50,0 |
| reglas janv1 `p3_notag` (banco de Jan) | 0.533 | 0.782 | – | 345 / – / 45,8 |
| fusio_v7d (base) | 0.477 | 0.674 | 14 / 17 | 208 / 17,8 s / 41,6 |
| ulises E5 (`a0`) | 0.450 | 0.678 | 15 / 17 | 220 / 18,7 s / 35,2 |
| ulises E6 (`a1`) | 0.465 | 0.670 | 14 / 17 | 220 / 18,3 s / 36,8 |
| ulises E7 (`t0`, contactos) | 0.497 | 0.660 | 12 / 17 | 203 / 19,9 s / 36,6 |
| ulises E7 + E9 (`t0` + `g3`) | 0.480 | 0.693 | 13 / 17 | 203 / 19,3 s / 38,4 |
| ídem con reglas `p3` + Smart Tagging como entrada | 0.480 | 0.692 | 13 / 17 | 202 / 19,3 s / 38,7 |
| ulises E10 (`t2` + `g3`, sin golpeos) | 0.333 | 0.695 | 13 / 17 | 227 / 17,4 s / 37,9 |

Lectura honesta:
- **En la ventana pública ulises no mejora a fusio_v7d** (0.33-0.50 frente a 0.48 según la variante; el macroF1 sí sube algo, 0.66-0.695 frente a 0.674). Que dos variantes casi idénticas (`t0` y `t2`, 13 posesiones predichas las dos) den 0.48 y 0.33 muestra lo poco que mide esta ventana. Con 17 posesiones, una posesión
  acertada o fallada mueve el spellF1 ≈ 0.03-0.06 (la propia organización da una banda de ruido de 0.04), así que
  la diferencia no es interpretable en ningún sentido. No se usa para decidir (RULES §2, issue #16).
- Lo que sí es llamativo, y ya estaba en los informes de Jan: **toda la familia con TCN queda por debajo de las
  reglas solas en esta ventana** (macroF1 0.67 frente a 0.78-0.86). La causa visible: la GT tiene 4 tramos de balón
  muerto (6, 20, 18 y 14 s) y los modelos con TCN solo detectan uno; dos de los perdidos (20 s y 14 s) tienen el
  balón visible todo el tiempo. Las reglas detectan 3-4.
- ¿Es un fallo sistemático o ruido de 4 tramos? Medido en el banco externo (`diag3.py`): los tramos D de 5-15 s con
  balón visible son el 21 % en IDSSE y el 6 % en Metrica; fusio_v7d los detecta en el 93 % (IDSSE) pero solo en el
  39 % en Metrica (n = 18); el stack sube a 99 % / 83 %. Es decir, el banco sí contiene el caso y ahí ulises mejora,
  pero en los dos tramos de la ventana pública falla igual que fusio_v7d. Con 2 casos no se puede concluir; queda
  como **riesgo abierto de dominio** (balón parado con balón visible), no como criterio de decisión.
- Las reglas `p3` con Smart Tagging (lo que usa `models/v7.py` de Jan en la entrega) son la vía de Jan para ese caso;
  no se puede validar en externos (no tienen Smart Tagging).



## Estado final (16-nov-2026, 11:00 UTC)

**Modelo `ulises` = TCN v7d de Jan (sin tocar) + reglas `p3_notag` + stack LightGBM `t2` + Viterbi con duraciones
mínimas y bono de balón muerto de las reglas.** Parámetros para el partido (pliegue `tots`): a = 0.35, b = 0,
d = 0.2, s_dur = 0.5. Modelo y parámetros: `model/ulises_stack_t2.txt`, `model/ulises_params.json`.

| banco de Jan v3, 10 partidos, fuera de pliegue | fusio_v7d | **ulises** | Δ pareado [IC95] | partidos que mejoran |
|---|---|---|---|---|
| spellF1 `label_acc` | 0.6963 [0.664, 0.728] | **0.7533 [0.717, 0.783]** | **+0.0570 [+0.0421, +0.0741]** | 10/10 |
| spellF1 `label_dur` | 0.6921 | **0.7306** | **+0.0386 [+0.0241, +0.0534]** | 10/10 |
| spellF1 `label_prov` | 0.4914 | **0.5406** | **+0.0493 [+0.0381, +0.0605]** | 10/10 |
| macroF1 `label_acc` | 0.8907 | 0.9114 | | |
| solo IDSSE (7) acc | 0.7052 | 0.7718 | +0.0665 | 7/7 |
| solo Metrica (3) acc | 0.6755 | 0.7103 | +0.0348 | 3/3 |
| posesiones predichas / GT | 205 / 242 | 214 / 242 | | |
| **ventana pública BAR-ATH (informativo)** | 0.477 / macro 0.674 | 0.333 / macro 0.695 | no interpretable (17 posesiones) | |

### Resumen de decisiones
| aceptadas | descartadas |
|---|---|
| E5 stack sobre la TCN (+0.042) | E1 rejilla más ancha, E2 duración mínima de D aparte (M1), E3 coste de transición: ≈ 0 |
| E7 contactos por distancia (+0.022 con golpeos; +0.015 sin ellos) | E4 pesos libres de reglas: +0.007 solo, pero no se suma al stack |
| E9 bono de D de las reglas en el decodificador (neutro en acc, + en prov y Metrica) | E6 / E12 ponderar por posesión: no se distingue en acc (sí mejora prov) |
| E10 quitar los golpeos por robustez de dominio (−0.007 acc, + dur) | E8 peso general de las reglas en el decodificador: empeora |
| | E13 entrenar con `label_prov`: empeora acc y dur |

### Lo que NO está resuelto (leer antes de proponerlo para la entrega)
1. **La mejora del banco no se ve en los 5 min públicos.** No es criterio de decisión y con 17 posesiones es ruido,
   pero tampoco hay ninguna evidencia en el dominio objetivo de que ulises sea mejor que fusio_v7d.
2. **Balón muerto en el BAR-ATH:** los modelos con TCN (fusio_v7d y ulises) ven 5-9 puntos menos de D que las reglas
   en el partido, y en el banco coinciden. En la ventana pública fallan 3 de 4 tramos D. Las reglas solas (`p3` con
   Smart Tagging) puntúan más que toda la familia TCN en esa ventana (0.57 / macro 0.86). No se puede validar en
   externos. Es el riesgo más serio de toda la línea fusio/markius/ulises, no solo de este modelo.
3. **El banco es distinguible del concurso** (AUC adversarial 0.99). El balón del ATD se comporta distinto cerca de
   los jugadores (golpeos 3 % frente a 8-13 %). `pilota_acc` es una entrada de la TCN v7d y está desplazada.
4. **Dependencia nueva:** LightGBM 4.7 (está en `.venv`, no en el venv del banco ni en `requirements-submission.lock`).
   La reproducibilidad byte a byte del CSV entre máquinas no está comprobada (solo determinismo en esta VM con
   `deterministic=True`, 2 hilos; las probabilidades se redondean a 5 decimales antes de decodificar).
5. El stack solo se entrena con IDSSE (7 partidos). SkillCorner no entra porque sus probabilidades de la TCN no son
   fuera de pliegue.
6. Pregunta de convención (E12/E13): si el equipo cree que la GT se parece más a `label_prov`, las variantes `t3` o
   `p2` son mejores en esa etiqueta; con la regla actual (`label_acc` principal) no entran.

### Ideas que quedan sin probar
- Features de contacto relativas a cada partido (umbrales por cuantil) para recuperar los golpeos de forma robusta.
- Pasar `ulises` por el arnés oficial de Jan (`bench/harness.py`) como un modelo más, para que salga en su informe.
- EM sin etiquetas del prior de D en el partido (idea "perfil bayesiano" de `lecciones_rawtree.md`), dirigida al
  riesgo 2; zona gris con el issue #2.
- Ensemble de `t2` y `t3` promediando log-probabilidades.

### Cómo reproducir
```
cd modelos_ulises/ulises
P=/workspace/shared/venvs/bench/bin/python ; V=../../.venv/bin/python
$P lab.py                                   # baseline fusio_v7d (results/base_fusio.json)
$V stack.py t2 0 tcn,jv,raw,touch           # entrena el stack (8 modelos, ~45 min) -> cache/stack_t2/
$P exp_stack.py t2 g3                       # evaluación anidada -> results/stack_t2_g3.json
MTAH_DATA_DIR=~/mtah-2027-possession/data $P match_entry.py     # entradas del BAR-ATH
$V make_csv.py t2 stack_t2_g3 tcn,jv,raw,touch                  # out/ulises.csv + sha256
```
`out/ulises.csv`: sha256 `02e5bce8…cb8e`, válido según `evaluation/validate_submission.py`; 227 posesiones,
17,4 s de media, 37,9 % D.


## Reentreno con el banco propio v8.2 (16-nov, 20:50 UTC)

Misma arquitectura y mismo protocolo, sobre `~/mtah_data/bench_v8` (degradación propia + SkillCorner con los equipos
corregidos; ver `../dataset_v8/REPORT_DATASET.md`). TCN y stack entrenados en el PC de Ulises. Un primer intento
(v8.1) falló porque los equipos de SkillCorner están invertidos en el banco del equipo
(`../docs/aviso_skillcorner_equipos.md`); sus resultados quedan en `results_v8_sc_invertido/`.

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

Configuración elegida (pliegue `tots`): a = 0,35, d = 0, s = 0,25. Modelo en `model_v8/`; CSV `out/ulises_v8.csv`.
Inferencia única: `predict.py` (reproduce byte a byte `out/ulises.csv` y `out/ulises_v8.csv` en la VM; x86 sin comprobar).

**Decisión: no se sustituye el modelo entrenado con el banco de Jan por el de v8.2 por mérito demostrado**: no hay
evidencia en el dominio objetivo a favor de ninguno de los dos. Quedan como dos candidatos; la elección es del equipo.


## Experimentos tras el reentreno (16-nov, 21:00–21:40 UTC)

| id | qué | resultado | decisión |
|---|---|---|---|
| E14 | Decodificador ampliado (`exp_stack.py t2 g4`, 108 configuraciones): mezcla del stack con la TCN (`t` = 0 / 0,1 / 0,2) y duración mínima propia para balón muerto (`sd`, la idea M1 de v9m1), sobre la rejilla g3 | Banco de Jan: elige exactamente las mismas configuraciones que g3 (0,7533, diferencia 0,0000). Banco v8.2: 0,7232 frente a 0,7219 (+0,001; `label_dur` −0,007; `label_prov` +0,008) | **Descartada**: no hay margen en el decodificador, tampoco con la idea de v9m1 |
| E15 | Promediar `model_v3` y `model_v8` (media de log-probabilidades del stack) en Metrica, el único conjunto común | `label_acc` / `label_dur` / `label_prov`: model_v3 0,710 / 0,665 / 0,630; model_v8 0,679 / 0,615 / 0,576; media 0,703 / 0,644 / 0,604 | **Descartada**: la media no supera a model_v3 |
| E16 | **Los dos modelos en Metrica degradada con v8** (`metrica_v8.py`; 3 partidos × 3 semillas en `~/mtah_data/bench_v9`), para compensar el sesgo de E15 | `label_acc` / `label_dur` / `label_prov`: model_v8 0,667 / 0,604 / 0,567; model_v3 0,623 / 0,587 / 0,558 (model_v8 gana en 3/3, 3/3 y 2/3 partidos). Fusio con la TCN v8: 0,665; con la v7d: 0,602; reglas solas: 0,472 | Informativa: **cada modelo gana en la degradación con la que se entrenó** |

**Lectura conjunta de E15 y E16 (Metrica, `label_acc`).**

| | Metrica con la degradación de Jan | Metrica con la degradación v8 | media |
|---|---|---|---|
| `model_v3` | 0,710 | 0,623 | 0,666 |
| `model_v8` | 0,679 | 0,667 | 0,673 |

Fuera de su degradación, `model_v8` pierde 0,031 y `model_v3` pierde 0,044: prácticamente un empate, con `model_v8`
algo más robusto. La pregunta que decide es **cuál de las dos degradaciones se parece más al partido**, y eso sí está
medido sin etiquetas: la v8 (distancia de descriptores 0,42 → 0,15; AUC banco-vs-partido 0,90 → 0,74 en IDSSE y
0,998 → 0,856 en SkillCorner). Con esa premisa, `model_v8` es el candidato mejor fundado de los dos. Sigue siendo un
argumento indirecto: no hay medida de acierto en el partido.
Consecuencia para el futuro: la idea 8 (entrenar con las dos degradaciones a la vez) sube al primer puesto.

**Lo que E15 enseña sobre los dos candidatos.** En Metrica `model_v3` gana a `model_v8` en los 3 partidos y con las 3
etiquetas. Ojo con leerlo como veredicto: Metrica está degradada con el esquema de Jan, el mismo con que se entrenó
`model_v3`, así que la comparación favorece a `model_v3`. A favor de `model_v8` solo hay señales sin etiquetas en el
partido (balón muerto y acuerdo con las reglas). Ninguna de las dos cosas es una medida de acierto en el partido.

## Ideas de mejora tras v8.2 (ordenadas por relación potencial / coste)

**Sin reentrenar la TCN**
1. **Promedio simétrico en la TCN al predecir** (como ya hace el stack): pasar el partido también con los equipos
   intercambiados (`permutacio_intercanvi` de `bench/train_tcn.py`) y promediar. Quita el sesgo local/visitante que
   la TCN hereda de los datos. Coste: inferencia de los 8 pliegues (~15 min) + stack (37 min VM / 12 min PC). Sin probar.
2. ~~Degradar Metrica con v8 para comparar `model_v3` y `model_v8`~~ — hecho, E16.
3. Features de contacto relativas al partido (umbrales por cuantil) para recuperar los golpeos sin desplazamiento
   de dominio. Coste: stack (37 min). Pendiente desde la iteración 4.
4. Pasar `ulises` por el arnés de Jan (`bench/harness.py`) para que salga en la tabla del equipo con su "domini".
5. Ajuste sin etiquetas del prior de balón muerto en el partido (EM). Zona gris con las reglas: consultar antes.

**Reentrenando (PC de Ulises: TCN ~2 h, stack ~12 min)**
6. **Stack entrenado también con SkillCorner** (27 partidos en vez de 7). Hoy no entra porque sus probabilidades de la
   TCN no son fuera de pliegue. Hace falta repartir SkillCorner en pliegues (p. ej. 4 pliegues de 5 partidos, +1 h de
   TCN). Con los equipos ya corregidos es la idea con más potencial: el stack es la pieza que más aporta y la que
   menos datos tiene.
7. **TCN del equipo con SkillCorner corregido en el banco de Jan** (sin tocar la degradación). Beneficia a v9m1 y a
   fusio directamente. Es trabajo de Jan o hay que copiar su banco a local e invertir `players_team`.
8. **Entrenar con las dos degradaciones a la vez** (banco de Jan + v8.2 como aumentación): hace al modelo menos
   dependiente de que una degradación concreta se parezca al partido, que es justo lo que no sabemos.
9. Varias semillas de la TCN y promediar probabilidades (menos varianza; 2 h por semilla).
10. Pérdida con más peso en las fronteras o una salida auxiliar de "cambio de posesión" (Marc tiene `boundary_tcn`
    en `main`): ataca los cambios local↔visitante que se pierden (47 % en el diagnóstico de v9m1).
11. Dataset: que el balón reaparezca en movimiento antes del saque (lo que dejó sin resolver la v9).


## Versión v10 (preparada el 16-nov por la noche; NO sustituye a `model_v3` ni a `model_v8`)

Junta las ideas 1, 2 y 3 de la lista anterior en una versión nueva con sus propios directorios (`~/mtah_data/v10`,
`cache_v10*`, `results_v10*`, `model_v10*`).

| pieza | qué cambia | dónde |
|---|---|---|
| Banco v10 | las dos degradaciones a la vez: semillas 0-2 = banco v8.2 tal cual; semillas 3-5 = degradación de Jan (IDSSE, SkillCorner con los equipos corregidos) y Metrica degradada con v8. Son enlaces, no copias | `dataset_v8/build_v10.py` |
| TCN (idea 1) | entrena con las 6 semillas (cada época sigue tomando una semilla al azar por partido, así que no tarda más) | `TCN_LLAVORS=0,1,2,3,4,5` |
| Pliegues de SkillCorner (idea 2) | 4 pliegues extra que dejan fuera 5 SkillCorner cada uno, para tener sus probabilidades fuera de pliegue | `TCN_PLECS_SC=4` |
| TCN simétrica (idea 3, E17) | media de la predicción directa y de la predicción con los equipos intercambiados. Se guardan las dos | `TCN_SIM=1` → `tcn_probs_sim/`; `predict.py` lo aplica si `params.json` trae `tcn_simetrica` |
| Stack con SkillCorner (idea 2) | variante `t2sc`: entrena con los 7 IDSSE + 20 SkillCorner | `ULISES_SC=1` |

Comprobado en la VM antes de dejarlo: los 12 pliegues se montan bien y cada SkillCorner cae en un pliegue; la
predicción simétrica funciona (en un pliegue de v8.2 la exactitud por frame pasa de 0,8948 a 0,9019); el stack con
SkillCorner entrena y predice (prueba corta con `ULISES_PROVA=1`); `predict.py` sigue dando el mismo sha256 para `model_v8`.

**Noche en el PC** (`noche_pc.ps1`, ~5–6 h): TCN v10 (12 pliegues + final) y cuatro stacks: {`t2`, `t2sc`} × {normal, simétrica}.
**Noche en la VM** (`noche_vm.sh`, ~45 min): E17 aislado sobre el banco v8.2 (sin reentrenar la TCN) → `results_v8_sim/`.
**Mañana en la VM**: `bash evalua_v10.sh` da, para cada variante, fusio base y los dos stacks en las semillas 0-2
(comparable con `results_v8/`: fusio 0,694, `ulises` 0,722) y en las semillas 3-5.

**Criterio de aceptación (fijado antes de ver resultados).** Una variante v10 sustituye a `model_v8` como candidata
solo si, frente a `model_v8`: (a) no baja en las semillas 0-2 con ninguna de las tres etiquetas, (b) en las semillas
3-5 sube en IDSSE respecto a `model_v8` aplicado ahí, y (c) en Metrica no baja con ninguna de las dos degradaciones
(E15/E16). `t2sc` frente a `t2`, y simétrica frente a normal, con el mismo criterio pareado (intervalo que no cruce el
cero o al menos 7 de 10 partidos). Si varias pasan, la más simple.

## Idea 5 — fronteras: qué hay y qué se puede hacer

Lo que ya existe: la TCN de Jan ya pondera ×3 los ±12 frames alrededor de cada cambio (`PES_FRONTERA`, `RADI_FRONTERA`,
fijados por definición, sin ajustar). El script de Marc `starter/train_tcn_boundary.py` (rama `main`) es el origen de
esa idea. El diagnóstico de v9m1 dice que aun así se pierde el 47 % de los cambios local↔visitante.

| idea | qué es | coste | comentario |
|---|---|---|---|
| 5a | **Detector de cambios en el stack + penalización variable en el Viterbi.** Un segundo LightGBM binario ("¿hay un cambio de posesión en ±12 frames?") con las mismas 104 features; su probabilidad abarata la transición en el Viterbi en esos frames. `lab.viterbi` ya admite una penalización por frame (`xpen`) | stack (37 min VM / 12 min PC) + rejilla de 1 parámetro; **no reentrena la TCN** | la más barata y la que ataca directamente los cambios cortos; empezar por aquí |
| 5b | Salida auxiliar en la TCN: además de H/A/D, predecir "cambio en ±k frames" (y su tipo). El stack recibe esa probabilidad como feature | TCN (2 h PC) + stack | multitarea clásica en segmentación de acciones; requiere tocar `models/tcn_v7.py` y la pérdida |
| 5c | Peso por tramo en la pérdida de la TCN (∝ 1/duración del tramo), para que una posesión corta pese como una larga, que es lo que hace spellF1 | TCN (2 h PC) | en el stack se probó (variantes `a0`/`a1`) y no ayudó; en la TCN está sin probar |
| 5d | Barrer `PES_FRONTERA` (3 → 5, 8) y `RADI_FRONTERA` (12 → 6, 25) | TCN (2 h PC) por valor | lo más simple, pero cada punto cuesta un entreno |
| 5e | Etiquetas suaves alrededor del cambio (±5 frames, el margen de sincronía que ya usa Jan entre eventos y tracking) | TCN (2 h PC) | evita castigar a la red por un desfase de 0,2 s que es ruido de las etiquetas |
| 5f | Pérdida de suavidad temporal (T-MSE de MS-TCN) para reducir cambios espurios | TCN (2 h PC) | el problema medido es de cambios perdidos, no espurios: prioridad baja |


## Resultados de la noche del 16 al 17-nov (E17 y versión v10)

TCN v10 (12 pliegues + final) y los cuatro stacks entrenados en el PC de Ulises; evaluación en la VM
(`evalua_v10.sh` → `evalua_v10.log`, `results_v10*`; comparaciones pareadas con `compara_res.py`).

**E17 — TCN simétrica sobre el banco v8.2, sin reentrenar la TCN** (`noche_vm.sh`, `results_v8_sim/`):

| | `label_acc` | `label_dur` | `label_prov` |
|---|---|---|---|
| fusio: simétrica − normal | +0,013 [+0,002, +0,023], 8/10 | +0,014 [+0,002, +0,025], 8/10 | −0,000, 5/10 |
| `ulises`: simétrica − normal | **0,732** (+0,010 [+0,003, +0,017], 9/10) | 0,707 (+0,004 [−0,003, +0,011], 8/10) | 0,518 (+0,005 [+0,000, +0,010], 8/10) |

Aceptada como mejora: sube con las tres etiquetas y en 9 de 10 partidos con la principal.

**Versión v10** (spellF1 `label_acc` / `label_dur` / `label_prov`; pliegues anidados, 10 partidos):

| variante | semillas 0-2 (= banco v8.2) | semillas 3-5 (IDSSE con degradación de Jan + Metrica con v8) |
|---|---|---|
| `model_v8` (referencia) | 0,722 / 0,703 / 0,513 | — |
| v10 `t2` | 0,726 / 0,714 / 0,513 | 0,739 / 0,715 / 0,519 |
| v10 `t2` simétrica | **0,735 / 0,718 / 0,528** | **0,751 / 0,716 / 0,536** |
| v10 `t2sc` (stack con SkillCorner) | 0,710 / 0,686 / 0,515 | 0,722 / 0,695 / 0,519 |
| v10 `t2sc` simétrica | 0,714 / 0,687 / 0,520 | 0,735 / 0,697 / 0,528 |

Decisiones dentro de v10 (pareadas):
- **Stack con SkillCorner (idea 2): descartado.** `t2sc` − `t2` = −0,016 [−0,025, −0,008], 2 de 10 partidos, en los dos
  grupos de semillas. Las etiquetas de SkillCorner no ayudan al stack (otra convención y otro tracker).
- **TCN simétrica (idea 3): aceptada.** +0,009 [+0,001, +0,016] en las semillas 0-2 y +0,012 [+0,006, +0,018] en las
  3-5; `label_prov` +0,015 y +0,018 (10/10 y 9/10).
- **Dos degradaciones (idea 1):** frente a `model_v8` en su mismo banco, v10 `t2` normal no se distingue (+0,004
  [−0,010, +0,018]); la ganancia viene de combinarla con la simetría.

**Lo que consigue la v10 simétrica frente a los dos modelos anteriores** (`label_acc`):

| conjunto | `model_v3` | `model_v8` | v10 `t2` simétrica |
|---|---|---|---|
| IDSSE, degradación de Jan (anidado) | 0,772 | — | 0,788 |
| IDSSE, degradación v8.2 (anidado) | — | 0,740 | 0,747 |
| Metrica, degradación de Jan | 0,710 | 0,679 | 0,705 |
| Metrica, degradación v8 | 0,623 | 0,667 | 0,665 |

Es decir: iguala o supera al especialista de cada degradación en IDSSE, y en Metrica queda a 0,005 y 0,002 del mejor
en cada caso, sin el hundimiento cruzado que tienen los otros dos (0,623 y 0,679).

**Aplicación del criterio de aceptación (el escrito el 16-nov, sin cambiarlo), para v10 `t2` simétrica frente a `model_v8`:**
- (a) semillas 0-2, ninguna etiqueta baja: **se cumple** (+0,013 [+0,001, +0,024] 8/10; +0,015 8/10; +0,015 8/10).
- (b) semillas 3-5, IDSSE por encima de `model_v8` aplicado ahí: **no se puede calcular como se escribió** (haría falta
  un stack de `model_v8` por pliegue, que no se guardó). Evidencia indirecta: 0,788 frente a 0,772 de `model_v3`, que
  es el especialista de esa degradación.
- (c) Metrica no baja con ninguna degradación: con la de Jan **se cumple** (0,705 / 0,656 / 0,617 frente a
  0,679 / 0,615 / 0,576); con la v8, `label_dur` y `label_prov` suben 0,001 y `label_acc` **baja 0,002**
  (0,665 frente a 0,667; por partido 0,742 / 0,685 / 0,568 frente a 0,748 / 0,683 / 0,570).

**Decisión.** A la letra, el criterio no queda cumplido entero: (b) no es calculable y (c) falla por 0,002 en una de
seis casillas, una diferencia muy por debajo del ruido de 3 partidos. **No declaro que la v10 sustituye a `model_v8`
por criterio**; la dejo empaquetada como tercera candidata y la decisión es del equipo. Mi lectura: es la mejor
fundada de las tres versiones de `ulises`, porque es la única que no depende de acertar qué degradación se parece al
partido.

Empaquetada en `model_v10_sim/` (configuración del pliegue `tots` de las semillas 0-2: a = 0,35, d = 0, s = 0,25;
`tcn_simetrica: true`). CSV `out/ulises_v10_sim.csv`, sha256 `b6809a0ca4c1038a00ce61a8cc554b36283736b4b3a01bd10021ff98e4c4a1e8`
(VM; **falta comprobarlo en el PC**). En el partido: 197 posesiones, 39,8 % de balón muerto. Ventana pública
(informativo, no decide): spellF1 0,492, macroF1 0,736 (`model_v8`: 0,492 / 0,729; `model_v3`: 0,333 / 0,695).


## Balón muerto: diagnóstico y corrección (17-nov, mañana)

**Punto de partida (informativo, ventana pública).** Las tres versiones de `ulises` predicen un 6 % de balón muerto en
la ventana, frente al 19,6 % real, y 9 posesiones donde hay 17. Esto solo se usó para saber dónde mirar; lo que sigue
se mide con los partidos externos y, en el BAR-ATH, sin etiquetas.

**E18 — dónde se pierde el balón muerto** (`diag_d.py`, v10 simétrica, `results/diag_d_E18.json`):

| | banco IDSSE | BAR-ATH (sin etiquetas) |
|---|---|---|
| balón muerto con balón visible: reglas / TCN / stack / final | 0,12 / 0,11 / 0,10 / 0,10 (semillas 0-2) | 0,148 / 0,154 / **0,090** / 0,086 |
| el stack mantiene el balón muerto de la TCN | 96–97 % | **89 %** |
| …cuando TCN y reglas coinciden y el balón se ve | 92 % | **77 %** |

El decodificador conserva el 98–99 % de lo que le da el stack: la pérdida está en el stack. En los frames donde TCN y
reglas dicen balón muerto con el balón visible, la velocidad del balón tiene mediana **1,41 m/s en el partido**,
0,38 m/s en el banco v8.2 y 0,00 m/s con la degradación de Jan (82 % de esos frames por debajo de 0,3 m/s en el
banco de Jan, 43 % en v8.2, 24 % en el partido). En el partido el balón muerto visible se mueve (se lleva al punto de
saque); en el banco está quieto, y el stack aprendió "balón visible en movimiento = en juego".

**E19 — stacks sin el atajo** (sobre la TCN v10 simétrica; `results_v10_sim_s012`, `_s345`; `e19.log`):

| variante | banco s0-2 (Δ frente a t2) | banco s3-5 | balón muerto visible en el BAR-ATH (stack) | stack mantiene D de la TCN en el BAR-ATH |
|---|---|---|---|---|
| t2 (actual) | 0,735 | 0,751 | 0,090 | 0,891 |
| t2nv: sin velocidad ni aceleración del balón | 0,731 (−0,004 [−0,006, −0,002], 0/10) | 0,744 (−0,007, 1/10) | 0,119 | 0,929 |
| t2tj: solo TCN + reglas | 0,721 (−0,014, 1/10) | 0,733 (−0,018, 1/10) | 0,123 | 0,938 |

Quitar la velocidad corrige más de la mitad del desajuste en el partido, pero cuesta 0,004–0,007 en el banco en casi
todos los partidos. Por la regla del proyecto (un cambio no puede empeorar en el banco) **no se acepta**. El banco no
puede premiar esta corrección porque en él no existe balón muerto visible en movimiento: hay que arreglar el banco.

**C11 / banco v11 — balón muerto visible en movimiento** (`dataset_v8/carry11.py`; detalle en `REPORT_DATASET.md`).
Post-proceso sin etiquetas sobre IDSSE v8.2: en cada tramo visible con el balón casi quieto (velocidad suavizada en 1 s
< 0,8 m/s durante ≥ 3 s), con probabilidad 0,8, el balón hace hasta 4 excursiones de ida y vuelta (≤ 10 s cada una,
velocidad media lognormal con mediana 2 m/s) y queda quieto el último segundo y medio. Calibrado contra el BAR-ATH:
en los frames D visibles del banco la velocidad pasa de mediana 0,42 m/s y 36 % < 0,3 a 1,25 m/s y 19 % < 0,3
(objetivo: 1,41 m/s y 24 %). La degradación de Jan no se corrige con este método (su balón muerto visible son trozos
cortos y exactamente quietos): v11 usa solo la mitad v8.2. La TCN no se reentrena (v10 simétrica); solo el stack.

**E20 — stacks entrenados en v11** (`e20.sh` → `e20.log`): t2 y t2nv entrenados con v11; evaluados en v11, en el banco
v10 semillas 0-2 (sin C11; con `xeval.py`, pliegues guardados) y en el partido sin etiquetas. Criterio, fijado antes
de verlos: se acepta el stack v11 t2 si en el banco sin C11 no baja frente al stack de `model_v10_sim` con ninguna
etiqueta (intervalo pareado que no quede por debajo de cero) y en el BAR-ATH mantiene más balón muerto de la TCN que
el actual (> 0,891).

Resultado (`e20.log`, 17-nov 09:57 UTC):

| | banco v11 (s0-2) | banco v10 sin C11 (s0-2), Δ frente a `model_v10_sim` | BAR-ATH: D visible stack / final | stack conserva D de la TCN |
|---|---|---|---|---|
| `model_v10_sim` (stack actual) | — | 0,735 | 0,090 / 0,086 | 0,891 |
| stack v11 t2 | 0,724 | 0,726 (−0,009 [−0,014, −0,004], 2/10; Metrica −0,020) | **0,128 / 0,129** | **0,922** |
| stack v11 t2nv | 0,720 | 0,722 (−0,013, 1/10) | 0,125 / 0,122 | 0,924 |

- En el partido, entrenar el stack con C11 hace lo que se buscaba: el balón muerto visible final sube de 0,086 a 0,129
  (TCN 0,154, reglas 0,148) y el stack conserva el 92 % del de la TCN. Además **ya no hace falta quitar la velocidad
  del balón**: t2 y t2nv se comportan igual en el partido, y t2 es mejor en el banco (+0,004, 8/10).
- En el banco anterior baja 0,009 (2/10), sobre todo en Metrica. **Por el criterio, no se acepta.**
- Confusión que impide atribuir la caída a C11: el stack v11 se entrenó con 3 semillas (solo la mitad v8.2) y el de
  `model_v10_sim` con 6 (las dos degradaciones). La v12 (las dos mitades con C11, 6 semillas) y el control E21 lo separan.

**E21 (control, VM):** stack t2 con la TCN v10 simétrica sin reentrenar, pero entrenado con las 6 semillas del banco v12
(las dos mitades con C11). Mismo criterio que E20. Aísla el efecto de C11 en el stack del de reentrenar la TCN.

Resultado (`e21.log`, 17-nov 10:53 UTC):

| | banco v10 sin C11, s0-2 (Δ frente a `model_v10_sim`) | ídem s3-5 | banco v12 (C11) s0-2 / s3-5 | BAR-ATH: D visible final | stack conserva D de la TCN |
|---|---|---|---|---|---|
| `model_v10_sim` | 0,735 / 0,718 / 0,528 | 0,751 / 0,716 / 0,536 | — | 0,086 | 0,891 |
| **E21** | 0,734 / 0,716 / 0,528 (−0,001 [−0,002, +0,001] 4/10; −0,002 [−0,004, +0,001] 3/10; +0,000 5/10) | 0,750 / 0,715 / 0,535 (−0,001; **`label_dur` −0,002 [−0,003, −0,000]**, 2/10; −0,001) | 0,734 / 0,748 | **0,122** | **0,928** |

Con las 6 semillas desaparece la caída de E20: era falta de datos, no la corrección. **Aplicando el criterio de E20
(banco v10 sin C11, semillas 0-2): se cumple** — las tres etiquetas tienen el intervalo cruzando el cero — y en el partido
el stack conserva el 92,8 % del balón muerto de la TCN (> 0,891). En las semillas 3-5, que el criterio no incluía,
`label_dur` baja 0,0018 con un intervalo que roza el cero por debajo (−0,0001); lo dejo dicho.

**Decisión: E21 aceptado.** Nuevo candidato `model_v10c11_sim` = la misma TCN que `model_v10_sim` (fichero idéntico) +
el stack E21. CSV `out/ulises_v10c11_sim.csv`, sha256 `46e89651b287416a8867149ecfcabaf13918a9f1c82310ded9021f49145d9717`
(VM; pendiente en el PC). En el partido: 191 posesiones, 42,0 % de balón muerto (antes 197 y 39,8 %).
Ventana pública (informativo, no decide): spellF1 0,492 (igual), macroF1 0,736 → **0,837**, acierto por frame
0,817 → 0,868, balón muerto en la ventana 6,1 % → 11,1 % (real 19,6 %).



## Versión v12 (preparada el 17-nov, ~09:30 UTC; pendiente de lanzar en el PC)

Igual que la v10 (dos degradaciones, 12 pliegues de la TCN, predicción simétrica), pero con el balón muerto visible en
movimiento (C11) en IDSSE en las dos mitades: semillas 0-2 = banco v11; semillas 3-5 = degradación de Jan + C11
(`~/mtah_data/bench_v12j`; en esa degradación C11 solo mueve una parte, ver `REPORT_DATASET.md`). SkillCorner y Metrica
como en v10. Motivo: la TCN v10 también usa el atajo — con los mismos modelos, en los frames de balón muerto que C11 pone
en movimiento su acierto baja de 0,946 a 0,875 (probabilidad media de D de 0,88 a 0,77), sin cambiar los falsos D (0,029).

- Banco: `dataset_v8/build_v12.py` (enlaces). Script del PC: `~/mtah_data/v12/pc/v12_pc.ps1` (TCN + stacks t2 y t2nv).
- Evaluación en la VM: `evalua_v12.sh` (v12 en su banco; frente a E20 en las semillas 0-2; modelo completo en el banco
  v10 sin C11 con `tcn_inf_otro.py` + `xeval.py`, frente a `model_v10_sim`; empaquetado y comportamiento en el partido).

**Criterio de aceptación (fijado antes de ver resultados).** La v12 (con el stack que E20 haya preferido) sustituye a la
mejor candidata del momento si: (a) en el banco v10 sin C11, semillas 0-2, no baja frente a `model_v10_sim` con ninguna
de las tres etiquetas (intervalo pareado que no quede entero por debajo de cero); (b) en las semillas 0-2 de v12 (= v11)
no baja frente al candidato de E20; (c) en el BAR-ATH, sin etiquetas, el stack conserva al menos el 93 % del balón
muerto de la TCN y el balón muerto visible final no baja respecto al candidato vigente.


## Resultado de la v12 (17-nov, 14:30 UTC) — NO adoptada

TCN reentrenada con C11 en IDSSE (dos degradaciones, 12 pliegues, en el PC de Ulises) + stacks t2 y t2nv
(`evalua_v12.sh` → `evalua_v12.log`).

| comparación (spellF1 `label_acc` / `label_dur` / `label_prov`, Δ pareado) | v12 t2 | v12 t2nv |
|---|---|---|
| (a) banco v10 sin C11, s0-2, frente a `model_v10_sim` | −0,003 [−0,009, +0,002] / **−0,007 [−0,013, −0,002]** / −0,002 | −0,008 / −0,010 / −0,005 |
| (b) banco v12, s0-2, frente a E20 (lo que pedía el criterio) | +0,007 [−0,000, +0,014] / −0,003 / +0,006 | +0,002 / −0,006 / +0,003 |
| (b') banco v12, frente al vigente `model_v10c11_sim` (s0-2 / s3-5) | −0,002 / **−0,005 [−0,010, −0,001]** / −0,003 · −0,003 / −0,003 / −0,001 | −0,007 / −0,009 / −0,006 |
| (c) BAR-ATH sin etiquetas: stack conserva D de la TCN / D visible final | 0,921 / 0,120 (vigente: 0,928 / 0,122) | — |

**Decisión: la v12 no sustituye a `model_v10c11_sim`.** Falla (a) en `label_dur` y falla (c) (0,921 < 0,93 y el balón
muerto visible final no sube). Reentrenar la TCN con C11 no aporta sobre reentrenar solo el stack: en el partido la TCN
v12 predice lo mismo que la v10 (balón muerto visible 0,153 frente a 0,154) y en los bancos queda algo por debajo.
Empaquetada igualmente en `model_v12_sim/` (CSV sha256 `cd7a24ca…2cc1b`) solo como registro; no es candidata.


## E22 — ¿forzar el número de posesiones? (17-nov, tarde) — descartado

Idea (de Ulises): obligar al modelo a producir el número de posesiones correcto. Con los 5 min públicos no se puede (el
17 sale de la GT pública: sería ajustar con ella). Versión legítima, sobre `model_v10c11_sim` en el banco de referencia
del equipo (`e22_posesiones.py`): sesgo delta en todas las transiciones del Viterbi.

| variante | spellF1 | Δ frente a lo entregado |
|---|---|---|
| entregado (215 posesiones predichas de 242 reales de media) | 0,762 | — |
| techo: en cada partido, el delta que da su número REAL de posesiones (trampa) | 0,761 | −0,001 [−0,005, +0,004], 4/10 |
| legítimo: el delta que iguala el ritmo de posesiones de los otros IDSSE | 0,759 | −0,002 [−0,007, +0,002], 4/10 |
| mejor delta posible en cada partido (trampa, cota superior) | 0,769 | +0,007 |

Ni sabiendo el número real de posesiones mejora: el modelo pierde posesiones porque no las ve, no porque el decodificador
las fusione; forzar más cambios los coloca en sitios equivocados. Incluso la cota superior es +0,007. **Descartado.**


## E23 — reglas sin el retiming de la ventana pública (17-nov, tarde) → ACEPTADO: es el modelo entregado (Nausicaa)

La auditoría del equipo señaló que las reglas que usamos (`p3_notag`) incluyen el retiming "a mitad del vuelo" de los
cambios de posesión, que el equipo sacó de la ventana pública (`starter/principled_v1.py`: "The rule was read off those
same 5 minutes"). E23 rehace el stack entregado (E21) con las reglas `p2_notag` (= `p3_notag` con `retime=None`; cambian
el 3-4 % de los frames de las reglas) y todo lo demás igual: misma TCN, mismos datos, mismo entrenamiento.
Scripts: `e23_reglas_p2.py` (entradas), `stack.py` sobre el banco `v12t10p2`, `e23.sh` (evaluación).

**Criterio (fijado antes de ver resultados).** E23 sustituye a E21 si, frente a E21: (a) en el banco de referencia del
equipo (vista v3view) y (b) en nuestro banco v10 semillas 0-2, ninguna de las tres etiquetas baja con un intervalo
pareado entero por debajo de cero; y (c) en el BAR-ATH, sin etiquetas, el stack conserva más del 89 % del balón muerto de
la TCN (el mismo umbral que se exigió a E21). Si pasa, se entrega E23 y el documento deja de declarar el retiming.

**Resultado (17-nov, 19:10 UTC).** Diferencias E23 − E21, IC95 pareado, partidos mejores:

| banco | acc | dur | prov |
|---|---|---|---|
| (a) referencia del equipo (v3view) | 0,7607: −0,0010 [−0,0035, +0,0015] 3/10 | +0,0000 [−0,0028, +0,0024] | +0,0001 [−0,0011, +0,0012] |
| (b) v10 semillas 0-2 | +0,0000 [−0,0024, +0,0018] 6/10 | +0,0003 [−0,0014, +0,0020] | −0,0002 [−0,0021, +0,0013] |
| (info) v12 semillas 0-2 / 3-5 | +0,0000 / −0,0020 [−0,0041, +0,0002] | +0,0001 / −0,0014 | −0,0004 / −0,0010 |

(c) partido sin etiquetas: el stack conserva el 92,9 % del balón muerto de la TCN (E21: 92,8 %); D visible final 0,124
(E21 0,122). **Se cumplen (a), (b) y (c): E23 sustituye a E21.** Frente a v9m1 en el banco de referencia: 0,761
[0,720, 0,793], +0,059 [+0,042, +0,081], 10/10 (Metrica +0,034). Partido: 192 posesiones, 19,1 s, 42,1 % D.
Modelo `model_v10c11p2_sim` = `models/nausicaa/` del repo limpio; CSV sha256
`8c5523a89afa68d87764404e3ca85a1f381d99113f70b0cc7a5b905decce2d9d` (VM ARM, en el entorno de desarrollo y desde una
copia aislada del repo limpio; valida con `validate_submission.py`). Comprobado también en Windows x86 (clon limpio, 17-nov).
Nota: la primera cola de `e23.sh` se quedó esperando para siempre porque su `pgrep -f "stack.py t2 0"` encontraba la
línea de comandos del propio shell que la lanzó; se relanzó con `setsid`.

