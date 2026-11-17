# docs — development notes (Spanish)

Notes written during the challenge, kept as a record of how decisions were made. The technical document
(`../TECHNICAL_DOCUMENT.md`) is the summary in English; these files are the detail behind it.

| File | Content |
|---|---|
| `REPORT.md` | Experiment log E1–E23: hypothesis, acceptance criterion written before each run, result, decision |
| `proceso.md` | The stages of section 2 of the technical document, in chronological order |
| `arquitectura.md` | Architecture of the model, piece by piece |
| `aviso_skillcorner_equipos.md` | The SkillCorner team-code issue found in stage 5 |

During development the model was called `ulises` and its candidates were named after the benchmark they were
trained on. The delivered model, `models/nausicaa/`, is the candidate called `model_v10c11p2_sim` in these notes (experiment E23: the stack of E21, `model_v10c11_sim`,
retrained with rules that do not use the mid-flight retiming).
