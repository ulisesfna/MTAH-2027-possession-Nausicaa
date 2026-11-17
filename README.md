# Nausicaa — ball possession from tracking data

Submission of team **El Chato** to *More Than A Hack 2027* (match BAR–ATH). For every frame of the tactical video,
Nausicaa says which team has the ball (`H` home, `A` away), whether the ball is dead (`D`), or `X` outside the halves.
It reads **only the ATD tracking**.

**Start with the technical document:** [`TECHNICAL_DOCUMENT.md`](TECHNICAL_DOCUMENT.md) (PDF: [`TECHNICAL_DOCUMENT.pdf`](TECHNICAL_DOCUMENT.pdf)).

## Reproduce the submission file

Python 3.12, CPU only. The match data is not in this repository (it is not redistributable).

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python predict.py --data <data folder> --out submission.csv
```

Windows: `.venv\Scripts\pip ...` and `.venv\Scripts\python ...`.

- `<data folder>` is the folder of the competition data bundle that contains
  `tactical/BAR-ATH_tactical_atd.txt` and `tactical/BAR-ATH_tactical_atd.xml`. Alternatively set `MTAH_DATA_DIR`,
  or put the bundle in `./data`. No other file is read, and no label.
- Output: `submission.csv` and `submission.csv.sha256`. Expected sha256:
  `8c5523a89afa68d87764404e3ca85a1f381d99113f70b0cc7a5b905decce2d9d`
  (identical on Linux ARM and Windows x86).
- About one minute; one CPU thread, fixed seeds.
- Check the format with the organisers' validator: `python evaluation/validate_submission.py submission.csv`
  (from the challenge repository).

## Contents

| Path | What |
|---|---|
| `predict.py` | The single inference script: tracking → features → TCN → rules → stack → Viterbi → CSV |
| `feats_stack.py` | Features of the stacked model (numpy) |
| `models/nausicaa/` | Trained model: `tcn.pt` (TCN), `stack.txt` (LightGBM), `params.json` (decoder settings) |
| `requirements.lock` | Pinned dependencies of inference |
| `base/` | Shared code used at inference: data loading (`games.py`, `bench/adapter.py`), features (`bench/features.py`), TCN (`models/tcn_v7.py`), rules (`starter/`, `bench/tune_decoder.py`), decoder (`models/decoder.py`). Its constants for the training workstation are not used at inference |
| `training/` | Training and evaluation of the stack (nested protocol) and changes made to the TCN training script (`base_patches/`); `training/README.md` maps every number of the document to its script |
| `dataset/` | Degradation of the external matches so they resemble the provided tracking, and its label-free validation (`REPORT_DATASET.md`, in Spanish) |
| `results/` | Benchmark results (JSON, external matches only) |
| `fig/` | Figures of the technical document and the scripts that draw them and build the PDF |
| `docs/` | Development notes written during the challenge (Spanish): experiment log `REPORT.md`, architecture, process |

Training is published as code: it ran on the team's workstation, with the external data in `MTAH_SHARED` and
`MTAH_LOCAL`, and it is not bit-reproducible across machines. Inference is.

## Licence

MIT (see `LICENSE`). `base/games.py` comes from the challenge starter repository (MIT, © Metrica Sports).
External datasets are not included; see the technical document for their sources and licences.
