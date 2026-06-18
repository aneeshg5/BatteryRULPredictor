# Battery RUL Predictor

> Predicting State of Health and Remaining Useful Life of Li-ion batteries using
> deep learning and gradient-boosted trees — based on published SRA 2023 research.

![Python 3.11](https://img.shields.io/badge/python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688)
![Plotly Dash](https://img.shields.io/badge/Plotly%20Dash-2.15-3F4F75)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

## Why This Matters

Battery health prediction is mission-critical wherever cells can't be casually
swapped out — Tesla's BMS fleet telemetry, SpaceX's satellite and launch-vehicle power
systems, and EV powertrains at Rivian, Lucid, BMW, and Waymo all depend on knowing a
cell's remaining useful life before it fails. This project rebuilds a published SRA
2023 paper's approach from scratch with production-quality engineering, then tests
whether more sophisticated models actually do better — they don't, and that result is
itself the interesting finding (see [Results](#results)).

## Results

Cross-battery generalization (Approach 2: train on RW9, predict RW10/RW11/RW12) —
average RMSE across the three held-out batteries:

| Model                 | Avg RMSE (Approach 2) |
|------------------------|------------------------|
| **Paper DNN (ours)**   | **0.66%**              |
| LightGBM (ours)        | 0.68%                  |
| Attention (ours)       | 0.80%                  |
| LSTM (ours)            | 0.81%                  |
| Upgraded DNN (ours)    | 1.26%                  |
| Paper DNN (original)   | 1.49%                  |
| BLS-RVM (original)     | 1.55%                  |
| RNN + LSTM (original)  | 1.61%                  |

All six of our models beat every published baseline. The more interesting result is
*within* our own models: the exact paper-replica DNN — 2 hidden layers, no batch norm,
no dropout, no attention — beats every upgrade we threw at it, including a Transformer
encoder. LightGBM operating on the same per-row engineered features (rolling
mean/std, dv/dt) comes a close second. That points to the engineered features already
carrying the temporal signal that matters, with the bottleneck being the
data/feature relationship rather than model capacity — see `CHECKPOINTS.md` Phase 7
for the full investigation.

## Architecture

```
NASA RW9-RW12 (.mat)
        │
        ▼
preprocess.py  ──► SOH calculation, rolling/derivative features, min-max scaling
        │
        ▼
data/processed/*.parquet
        │
        ├──► PyTorch models (paper_dnn, upgraded_dnn, lstm, attention)
        │        via Trainer + MLflow tracking + Optuna tuning
        │
        └──► LightGBM (tree_trainer.py)
                 via mlflow.lightgbm

        │
        ▼
precompute_predictions.py ──► data/processed/predictions/*.parquet
        │
        ├──► Plotly Dash dashboard (scripts/serve.py)
        └──► FastAPI inference endpoint (uvicorn battery_rul.inference.api:app)
```

## Quick Start

```bash
git clone https://github.com/aneeshg5/battery-rul-predictor.git
cd battery-rul-predictor
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
brew install libomp && bash scripts/fix_macos_libomp.sh  # macOS only
python scripts/download_data.py
python scripts/train.py --model paper_dnn --approach 2   # repeat for other --model values
python scripts/precompute_predictions.py
python scripts/serve.py                                  # http://localhost:8050
```

## Dashboard

![Dashboard screenshot](docs/images/dashboard_screenshot.png)

Select a battery, model, and approach to see live voltage traces, an SOH gauge,
SOH-over-time, training curves, and a full model comparison table.

## Project Structure

```
battery-rul-predictor/
├── src/battery_rul/
│   ├── data/            # download, preprocess, PyTorch Dataset
│   ├── models/           # paper_dnn, upgraded_dnn, lstm, attention, lightgbm
│   ├── training/         # Trainer, Optuna tuning, LightGBM trainer
│   ├── evaluation/       # RMSE/MAE/R2, safety-ratio metrics, prediction precompute
│   ├── inference/         # Predictor + FastAPI app
│   └── dashboard/        # Plotly Dash app
├── notebooks/             # EDA and model-comparison notebooks
├── scripts/               # CLI entry points (download, train, tune, serve)
├── tests/                 # pytest suite
└── CHECKPOINTS.md          # full phase-by-phase engineering log
```

## Paper Reference

Chen, Ganti, Matsumura. "Predicting the Remaining Useful Life of
Lithium-Ion Batteries Using Machine Learning Techniques." SRA 2023.
Full transcription: [`docs/paper_transcription.md`](docs/paper_transcription.md).

## Author

Aneesh Ganti — [GitHub](https://github.com/aneeshg5)
