# Proceso seguido — borrador (16-nov-2026)

Relato cronológico de cómo se ha llegado al modelo actual, con las decisiones y por qué. Es un borrador para el
informe técnico: los números y los detalles están en los reports que se citan en cada fase; aquí solo va el hilo.
Las fases 1–3 son anteriores a esta sesión y están resumidas a partir de `docs/`; **Ulises: revísalas y complétalas**.

## Fase 1 — Reglas con decodificador (Viterbi v1, El Chato v2–v5)
- Idea: posesión = jugador más cercano al balón + suavizado temporal con un HMM/HSMM y secuencia de toques.
- Código: `solution/` (Viterbi v1), `solution_v2/` (HSMM v2–v5). Estado: `docs/progreso_chato_v5.md`.
- Qué se aprendió: el criterio de posesión no es único (convenciones events / ctrl2 / mid / tracking,
  `docs/criterio_posesion.md`) y un cambio solo vale si no empeora con ninguna.

## Fase 2 — Línea ML propia (ml_lr) y dataset externo
- Dataset externo unificado v2 / v2.1 (Metrica, SkillCorner, IDSSE): `docs/datos_externos.md`, `tools/`.
- Modelo ML por frame: `solution_ml/`, estado en `docs/progreso_chato_ml.md`.
- Regla que se fija aquí (respuesta de los organizadores, `docs/issues_organizadores.md`): nada se ajusta ni se
  elige con el BAR-ATH ni con los 5 min públicos.

## Fase 3 — Banco del equipo y candidata fusio_v7d (Jan)
- Banco: partidos externos degradados para parecerse al tracking del partido (degradación v1 → v3), 10 partidos de
  selección (7 IDSSE + 3 Metrica), selección anidada, bootstrap pareado entre partidos.
- fusio_v7d = TCN no causal (v7d) + reglas janv1 + Viterbi con duración mínima. Banco: 0,696 spellF1.
- Detalle y resto de modelos del equipo: `docs/comparativa_modelos.md`.

## Fase 4 — Modelo `ulises`: mejorar fusio_v7d (15–16 nov) → `ulises/REPORT.md`
1. **Protocolo antes que resultados**: mismo banco y misma selección anidada que Jan; línea base reproducida
   exactamente (0,6963).
2. **Diagnóstico**: dónde se pierde spellF1 (transiciones con balón visible, no huecos largos).
3. **Cambios en el decodificador (E1–E4)**: ganancias pequeñas o nulas → descartados.
4. **Modelo apilado (E5)**: LightGBM sobre las probabilidades fuera de pliegue de la TCN ± 2 s, medias móviles,
   reglas y features locales. Sustituye a las probabilidades de la TCN en el Viterbi. Aceptado.
5. **Iteraciones 2–5**: variantes de pesos y de features; secuencia de contactos por distancia (aceptada);
   features de golpeo (aceptadas y luego retiradas: la validación adversarial mostró que no se parecen entre
   banco y partido); bono a balón muerto donde las reglas dicen D (aceptado). Una afirmación del report ("el stack
   es un suavizado aprendido") se corrigió tras una ablación.
6. **Resultado en el banco**: 0,6963 → 0,7533 (+0,057, IC95 [+0,042, +0,074], mejora en los 10 partidos; Metrica,
   que no entra en ningún entrenamiento, +0,035).
7. **Prueba informativa en los 5 min públicos**: fusio_v7d 0,477; `ulises` 0,33–0,50 según variante; las reglas
   solas 0,53–0,57. El banco no predice lo que pasa en el partido → se pausa el modelo y se revisa el banco.

## Fase 5 — Corregir el banco (16 nov) → `dataset_v8/REPORT_DATASET.md`
1. **Qué se puede hacer**: inspeccionar el tracking del partido sí; ajustar contra la GT de los 5 min no. La
   validación se hace sin etiquetas: descriptores del tracking, AUC de un clasificador banco-vs-partido,
   comportamiento de modelos sonda.
2. **Diagnóstico del banco v3**: el balón desaparecía parado y fuera del campo (en el partido, en movimiento y
   cerca de un jugador, y reaparece golpeado); balón parado a velocidad exactamente cero; balón 3–10 veces más
   ruidoso; sin caídas de jugadores ligadas a los huecos de balón; sin detecciones con equipo erróneo.
3. **v8 (IDSSE)**: mecanismos C1–C9, calibrados con agregados del tracking del BAR-ATH. Una primera versión (v8a)
   ocultaba todo el balón parado y se corrigió (v8c). Distancia de descriptores 0,424 → 0,154; AUC 0,900 → 0,744.
4. **v8.1 (SkillCorner)**: con la degradación anterior se distinguía del partido casi al 100 % (AUC 0,998) por
   tener 9,5 jugadores visibles frente a 20,5. Usando las posiciones extrapoladas del proveedor más los mecanismos
   de v8: AUC 0,856. Cuatro variantes probadas, una aceptada.
5. **v9 (no adoptada)**: limitar la duración de los huecos de parón acerca el balón visible y la duración de los
   huecos, pero empeora la distancia global (0,117 → 0,136–0,141). Se deja documentada y desactivada.

## Fase 6 — Reentreno con el banco v8 (16 nov, en curso)
- Features y TCN (8 pliegues + final) con v8; la TCN se entrena en el PC de Ulises (Ryzen 7 5800U, 8 hilos) porque
  la VM tiene 2 núcleos. **A declarar**: los pesos no son idénticos bit a bit a los que saldrían en la VM.
- Después, en la VM: entradas del decodificador, fusio base, stack `ulises`, evaluación anidada en el banco v8 y
  predicción del partido (los 5 min públicos se miran una vez, como informe).
- Primer intento (v8.1): fusio 0,556 y la TCN confundiendo local y visitante. Causa: **los equipos de SkillCorner
  están invertidos respecto a las etiquetas en el banco del equipo** (`aviso_skillcorner_equipos.md`). No se había
  visto porque SkillCorner degradado era tan distinto de IDSSE que la TCN aprendía un convenio por fuente.
- Segundo intento (v8.2, equipos corregidos): la TCN vuelve al ~90 % y en el partido ya predice tanto balón muerto
  como las reglas (0,464 frente a 0,458; antes 0,409). `ulises` 0,722 en el banco v8.2 (+0,028 sobre fusio).
  En Metrica no mejora (0,710 → 0,679) y los 5 min públicos no distinguen (0,48–0,49).
- Decisión: el banco v8.2 se da por mejor banco; no hay evidencia de que el modelo reentrenado sea mejor en el
  partido. Quedan dos candidatos (`model_v3`, `model_v8`) y la elección es del equipo.
- Además: script de inferencia único `ulises/predict.py`, verificado byte a byte en la VM y en el PC (x86).

## Fase 7 — Versión v10 (noche del 16 al 17-nov)
- Pruebas rápidas previas: el decodificador no da más (E14); promediar los dos modelos no mejora (E15); en Metrica
  cada modelo gana en la degradación con que se entrenó (E15/E16). De ahí la idea: entrenar con las dos a la vez.
- v10 = TCN entrenada con las dos degradaciones + predicción simétrica (media con los equipos intercambiados) + stack.
  Se probó además entrenar el stack con SkillCorner.
- Resultado: la simetría se acepta (+0,010 en el banco v8.2, 9 de 10 partidos); el stack con SkillCorner se descarta
  (−0,016, 2 de 10); la v10 simétrica iguala o supera al especialista de cada degradación (IDSSE 0,788 y 0,747;
  Metrica 0,705 y 0,665) sin la caída cruzada de los otros dos modelos.
- Criterio de aceptación fijado antes: se cumple (a); (b) no era calculable como se escribió; (c) falla por 0,002 en
  una de seis casillas. No se declara sustitución; queda como tercera candidata (`model_v10_sim`).

## Excepciones y riesgos que el informe técnico debe contar
- La calibración de la degradación usa agregados del tracking del BAR-ATH (sin etiquetas), como la de Jan pero
  más amplia.
- Fuga de segundo nivel en el apilado: al puntuar un IDSSE, el stack se entrena con probabilidades de otros
  partidos producidas por TCN que sí vieron ese partido. Metrica no tiene esta fuga.
- Qué convención de etiquetas se parece a la GT del concurso sigue sin saberse.
- Ningún resultado en los 5 min públicos se ha usado para elegir nada.

## Fase 8 — Balón muerto (mañana del 17-nov)
- Diagnóstico (E18): en el partido el stack descartaba balón muerto visible que TCN y reglas veían. Causa: en los bancos
  el balón muerto visible está quieto y en el partido se mueve (mediana 1,4 m/s), y el stack lo había aprendido.
- Quitar la velocidad del balón (E19) corregía a medias y empeoraba el banco: descartado.
- Corrección del banco (C11): el balón muerto visible hace excursiones de ida y vuelta, calibradas sin etiquetas del partido.
- E20: stack entrenado con C11 solo en la mitad v8.2 → bien en el partido, peor en el banco (tenía la mitad de datos).
- E21: stack entrenado con C11 en las dos mitades → igual en el banco anterior y balón muerto visible en el partido de
  0,086 a 0,122. Aceptado: `model_v10c11_sim`.
- v12 (TCN también reentrenada con C11): no mejora sobre reentrenar solo el stack y falla su criterio; no adoptada.
- Candidato final de `ulises`: `model_v10c11_sim`. Reproducible byte a byte en ARM y x86, y con el código de la rama del
  equipo sin parches.

## Fase 9 — Reglas limpias y nombre final (tarde del 17-nov)
- La auditoría del equipo señaló que las reglas `p3_notag` incluyen el retiming "a mitad del vuelo", sacado de la ventana
  pública. E23: mismas piezas con `p2_notag` (sin retiming) y stack reentrenado. Neutro en los dos bancos y 92,9 % de
  balón muerto conservado: aceptado. Es el modelo entregado.
- El modelo pasa a llamarse **Nausicaa** (Odisea, canto VI: el primer juego de pelota de la literatura occidental) y se
  monta un repo limpio solo con él: `~/mtah_data/repo_nausicaa/`.
