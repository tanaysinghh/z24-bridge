# VibraDiagnose

AI-assisted structural health monitoring of the Z24 bridge (KU Leuven Progressive Damage Tests). Single-accelerometer damage-scenario classification with WaveNet, MiniRocket and InceptionTime.

## Layout

| Path | Contents |
|---|---|
| `data/` | Raw dataset (`data/z24`) and processed arrays (`data/processed`), not tracked |
| `src/preprocessing` | Channel selection, band-pass, windowing, z-score, setup-based split |
| `src/models` | WaveNet, MiniRocket, InceptionTime |
| `src/train` | One training script per model plus results summary |
| `models/` | Trained checkpoints, not tracked |
| `docs/` | Dataset, results and literature review documents |
| `notebooks/` | Exploration notebooks |

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='thanglexuan/Z24-dataset-processed', repo_type='dataset', local_dir='./data/z24')"
```

## Run

```bash
python -m src.preprocessing.pipeline
python -m src.train.train_minirocket
python -m src.train.train_wavenet
python -m src.train.train_inceptiontime
python -m src.train.summarize
```

See `docs/dataset.md` for every preprocessing decision and `docs/results_review2.md` for results.
