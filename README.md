# Battery Remaining Useful Life Predictor

## Overview

Remaining Useful Life (RUL) is how much usable life a battery has left before it needs
replacing. Accurate State of Health (SOH) estimates are critical wherever cells can't be
swapped out casually, like battery management systems, fleet telemetry, satellite and
launch vehicle power systems, and EV powertrains.

This project implements an end-to-end deep learning pipeline for SOH/RUL prediction. It
reimplements my SRA 2023 research paper's approach and benchmarks it against more
sophisticated architectures, including LSTM, Transformer-based attention, and LightGBM,
on real NASA battery degradation data. The pipeline covers data preprocessing and feature
engineering, model training and hyperparameter tuning with MLflow and Optuna, a FastAPI
inference service, and an interactive Plotly Dash dashboard. See Results below.

## Results

Cross-battery generalization (Approach 2: train on RW9, predict RW10/RW11/RW12).
Average RMSE across the three held-out batteries:

**Trained and evaluated in this project**

| Model            | Avg RMSE (Approach 2) |
|-------------------|------------------------|
| **Paper DNN**     | **0.66%**              |
| LightGBM          | 0.68%                  |
| Attention         | 0.80%                  |
| LSTM              | 0.81%                  |
| Upgraded DNN      | 1.26%                  |

**Published baseline (SRA 2023 paper)**

| Model            | Avg RMSE (Approach 2) |
|-------------------|------------------------|
| Paper DNN         | 1.49%                  |
| BLS-RVM           | 1.55%                  |
| RNN + LSTM        | 1.61%                  |

All six models beat every published baseline. The interesting result is within
our own models where the exact paper-replica DNN, with 2 hidden layers and no batch norm,
dropout, or attention, beats every upgrade we tried, including a Transformer encoder.
LightGBM, using the same per-row engineered features (rolling mean/std, dv/dt), comes
close second. This suggests the engineered features already carry the temporal
signal that matters, and the bottleneck is the data/feature relationship, not model
capacity.

## Getting Started

```bash
git clone https://github.com/aneeshg5/Battery-RUL-Predictor.git
cd Battery-RUL-Predictor
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
brew install libomp && bash scripts/fix_macos_libomp.sh  # macOS only
python scripts/download_data.py
python scripts/train.py --model paper_dnn --approach 2   # repeat for other --model values
python scripts/precompute_predictions.py
python scripts/serve.py                                  # http://localhost:8050
```

## Dashboard

![Dashboard demo](docs/images/dashboard_demo.gif)

Select a battery, model, and approach to see live voltage traces, an SOH gauge,
SOH-over-time, training curves, and a full model comparison table.

## Architecture and Project Structure

```
battery_rul/
├── data/         NASA RW9-RW12 (.mat) ──► preprocess.py ──► data/processed/*.parquet
├── models/       paper_dnn, upgraded_dnn, lstm, attention, lightgbm
├── training/     Trainer (MLflow + Optuna) for the torch models, tree_trainer.py for LightGBM
├── evaluation/   metrics.py + precompute_predictions.py ──► data/processed/predictions/*.parquet
├── inference/    Predictor + FastAPI app (uvicorn battery_rul.inference.api:app)
└── dashboard/    Plotly Dash app (scripts/serve.py)

notebooks/        EDA and model-comparison notebooks
scripts/          CLI entry points (download, train, tune, serve)
tests/            pytest suite
```

See [`docs/architecture.md`](docs/architecture.md) for the data pipeline, model and
training design decisions, and API rationale in detail.
