# Arquitectura del modelo `ulises` (16-nov-2026)

Modelo de posesión por frame para el BAR-ATH. Es **fusio_v7d de Jan con una pieza nueva en medio** (un modelo
apilado) y un cambio en la entrada del decodificador. Código en `modelos_ulises/ulises/`; registro de experimentos
en `ulises/REPORT.md`. Este documento describe qué hace cada pieza, no cómo se llegó a ella (eso está en
`proceso.md`).

## Esquema

```
tracking ATD (balón + jugadores, 25 fps)
        │
        ├──► [1] 35 features por frame ──► [2] TCN v7d ──► P_tcn(H, A, D)
        │                                                        │
        ├──► [3] reglas janv1 ──► etiqueta H/A/D, "clar", confianza física
        │                                   │                    │
        └──► features locales ──────────────┴────────► [4] stack LightGBM ──► P_stack(H, A, D)
                                                                               │
                              reglas dicen D ──► bono a D ──► [5] Viterbi con duración mínima
                                                                               │
                                                                    etiqueta final H / A / D / X
```
H = local, A = visitante, D = balón muerto, X = fuera de las partes (no se puntúa).

## [1] Features (`base/bench/features.py`, de Jan)
35 valores por frame, con el campo normalizado y orientado según la dirección de ataque:
- **Balón**: posición, si se ve, velocidad, aceleración, tiempo desde que se vio por última vez (log), última
  posición vista, distancia a la banda y al fondo.
- **Por equipo** (local y visitante): distancia al balón del jugador más cercano (`d1`) y del segundo (`d2`),
  velocidad relativa del más cercano respecto al balón, centro y dispersión del equipo, velocidad media, jugadores
  visibles.
- **Contexto**: parte, tiempo dentro de la parte.

## [2] TCN v7d (`base/models/tcn_v7.py`, de Jan)
- Red convolucional temporal **no causal** (cada frame ve pasado y futuro).
- Proyección de entrada 1×1 → 8 bloques residuales, cada uno con dos convoluciones de núcleo 3 dilatadas
  (dilataciones 1, 2, 4, …, 128), 64 canales, dropout 0,1 → proyección de salida 1×1 a 3 clases.
- Campo receptivo: 1021 frames (±510, unos ±20 s).
- Entrenamiento: 7 IDSSE + 20 SkillCorner degradados, 3 semillas de degradación, intercambio de equipos como
  aumentación, etiqueta `label_acc`, peso ×3 en los ±12 frames alrededor de cada cambio de posesión, hasta 7 épocas
  con parada temprana.

## [3] Reglas janv1 `p3_notag` (`base/models/`, de Jan)
Etiqueta determinista por frame según quién controla físicamente el balón (distancia y velocidad relativa), sin
Smart Tagging. Da además `clar` (si la regla está segura) y una confianza física entre 0 y 1.

## [4] Stack (`ulises/stack.py`) — la pieza nueva
- **Qué es**: un LightGBM multiclase (H/A/D), 250 rondas, 31 hojas, que corrige las probabilidades de la TCN
  mirando también las reglas y el tracking local.
- **Entrada: 104 features por frame**, en cuatro grupos:
  - `tcn` — log-probabilidades de la TCN en el frame y desplazadas ±5, ±12, ±25, ±50 frames (hasta ±2 s); medias
    móviles de 1, 3 y 6 s; frames desde/hasta el último/próximo cambio del argmax de la TCN.
  - `jv` — etiqueta de las reglas (one-hot) en el frame y en ±12, ±25; medias móviles de 1 y 3 s; frames
    desde/hasta su último/próximo cambio; `clar` y confianza física.
  - `raw` — 14 features del tracking (balón presente, velocidad, aceleración, tiempo sin verse, `d1`, `d2` y
    velocidad relativa de cada equipo, distancia a banda y fondo, jugadores visibles), la diferencia de `d1` entre
    equipos con desplazamientos y medias, y el mínimo de `d1` en ventanas de 0,5 y 2 s.
  - `touch` — secuencia de contactos por distancia: tiempo desde/hasta el último/próximo frame en que cada equipo
    tuvo el balón a menos de 1 m y de 2 m, y qué equipo tuvo el último y tendrá el siguiente contacto.
- **Lo que se quitó**: las features de golpeo (contacto + aceleración > 15 m/s²). Mejoraban el banco, pero la
  validación adversarial mostró que no se distribuyen igual en el banco y en el partido.
- **Entrenamiento**: solo IDSSE, con las probabilidades de la TCN **fuera de pliegue** (la TCN que las produjo no
  vio ese partido); un modelo por IDSSE dejado fuera y uno con los 7 para Metrica y para el partido. Aumentación
  por intercambio de equipos y promedio simétrico al predecir (se predice también con los equipos cambiados y se
  promedia).

## [5] Decodificador (`ulises/exp_stack.py`, sobre `models/decoder.py` de Jan)
- Viterbi sobre un HMM con **duración mínima** por estado; las transiciones y duraciones salen de las etiquetas
  de los partidos de entrenamiento.
- Emisión por frame: `a · (log P_stack − log prior) + d · [las reglas dicen D]` en la clase D.
- Parámetros elegidos en el banco (selección anidada): `a = 0,35`, `d = 0,2`, escala de duración `s = 0,5`. El
  peso directo de las reglas (`b`) queda en 0: su información ya entra por el stack y por el bono a D.
- Se decodifica cada parte por separado.

## Qué cambia respecto a fusio_v7d
| | fusio_v7d | `ulises` |
|---|---|---|
| Probabilidades que entran al Viterbi | las de la TCN | las del stack |
| Papel de las reglas | término de emisión con peso `w`, `lam` | entrada del stack + bono a D |
| Tracking local en la decisión | solo a través de la TCN | también directo, en el stack |

## Resultado y estado
- Banco v3 (10 partidos, selección anidada): spellF1 0,696 → **0,753** (+0,057, IC95 [+0,042, +0,074], mejora en
  los 10). Metrica, que no entra en ningún entrenamiento: +0,035.
- En los 5 min públicos (solo informativo) no mejora a fusio_v7d, y las reglas solas quedan por encima de ambos.
  Por eso se corrigió el banco (`../dataset_v8/REPORT_DATASET.md`) y se está reentrenando esta misma arquitectura
  con el banco v8. Ese resultado todavía no está.
- Ficheros del modelo entrenado con el banco v3: `ulises/model/ulises_stack_t2.txt` (stack) y
  `ulises/model/ulises_params.json` (decodificador); la TCN es `tcn_v7d_weights.pt` de Jan.
