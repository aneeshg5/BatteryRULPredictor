"""Precompute actual-vs-predicted SOH and training curves for the dashboard.

Loads each already-trained model from MLflow (most recent run per name) instead of
retraining, runs inference over a stride-subsampled trace of each battery, and writes
parquet files to data/processed/predictions/ so the dashboard never runs live inference.

python scripts/precompute_predictions.py
"""

import logging

import mlflow.lightgbm
import mlflow.pytorch
import pandas as pd
from mlflow.tracking import MlflowClient

from battery_rul.config import PREDICTIONS_DIR, PROCESSED_DIR
from battery_rul.data.preprocess import FEATURE_COLUMNS
from battery_rul.evaluation.predictions import predict_battery, predict_battery_lightgbm
from battery_rul.training.trainer import get_default_device

logger = logging.getLogger(__name__)

PAPER_FEATURE_COLUMNS = FEATURE_COLUMNS[:5]
TARGET_POINTS = 5000

RUN_CONFIGS = [
    {
        "model": "paper_dnn",
        "approach": 1,
        "run_name": "paper_dnn_approach1",
        "feature_columns": PAPER_FEATURE_COLUMNS,
        "input_mode": "flat",
        "batteries": ["RW9"],
    },
    {
        "model": "paper_dnn",
        "approach": 2,
        "run_name": "paper_dnn_approach2",
        "feature_columns": PAPER_FEATURE_COLUMNS,
        "input_mode": "flat",
        "batteries": ["RW10", "RW11", "RW12"],
    },
    {
        "model": "upgraded_dnn",
        "approach": 2,
        "run_name": "upgraded_dnn_approach2",
        "feature_columns": FEATURE_COLUMNS,
        "input_mode": "flat",
        "batteries": ["RW10", "RW11", "RW12"],
    },
    {
        "model": "lstm",
        "approach": 2,
        "run_name": "lstm_approach2",
        "feature_columns": FEATURE_COLUMNS,
        "input_mode": "sequence",
        "batteries": ["RW10", "RW11", "RW12"],
    },
    {
        "model": "attention",
        "approach": 2,
        "run_name": "attention_approach2",
        "feature_columns": FEATURE_COLUMNS,
        "input_mode": "sequence",
        "batteries": ["RW10", "RW11", "RW12"],
    },
]

LIGHTGBM_RUN_CONFIG = {
    "model": "lightgbm",
    "approach": 2,
    "run_name": "lightgbm_approach2",
    "feature_columns": FEATURE_COLUMNS,
    "batteries": ["RW10", "RW11", "RW12"],
}


def latest_run_id(client: MlflowClient, run_name: str) -> str:
    """Find the most recent MLflow run with the given run name (re-runs reuse names)."""
    runs = client.search_runs(
        experiment_ids=["0"],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No MLflow run found with name {run_name!r}")
    return runs[0].info.run_id


def save_history(client: MlflowClient, run_id: str, run_name: str) -> None:
    """Save per-epoch train/val RMSE history for the training-curve chart."""
    train_history = client.get_metric_history(run_id, "train_rmse")
    val_history = client.get_metric_history(run_id, "val_rmse")
    df = pd.DataFrame(
        {
            "epoch": [m.step for m in train_history],
            "train_rmse": [m.value for m in train_history],
            "val_rmse": [m.value for m in sorted(val_history, key=lambda m: m.step)],
        }
    )
    df.to_parquet(PREDICTIONS_DIR / f"{run_name}_history.parquet", index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_default_device()
    client = MlflowClient()

    for cfg in RUN_CONFIGS:
        run_id = latest_run_id(client, cfg["run_name"])
        logger.info("Loading %s from run %s", cfg["run_name"], run_id)
        model = mlflow.pytorch.load_model(f"runs:/{run_id}/model").to(device)
        save_history(client, run_id, cfg["run_name"])

        for battery in cfg["batteries"]:
            parquet_path = PROCESSED_DIR / f"{battery}.parquet"
            n_rows = len(pd.read_parquet(parquet_path, columns=["soh"]))
            stride = max(1, n_rows // TARGET_POINTS)
            result = predict_battery(
                model,
                parquet_path,
                cfg["feature_columns"],
                device,
                cfg["input_mode"],
                stride=stride,
            )
            out_name = f"{cfg['model']}_approach{cfg['approach']}_{battery}.parquet"
            out_path = PREDICTIONS_DIR / out_name
            result.to_parquet(out_path, index=False)
            logger.info("Wrote %s (%d points, stride=%d)", out_path, len(result), stride)

    cfg = LIGHTGBM_RUN_CONFIG
    run_id = latest_run_id(client, cfg["run_name"])
    logger.info("Loading %s from run %s", cfg["run_name"], run_id)
    lgbm_model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")

    for battery in cfg["batteries"]:
        parquet_path = PROCESSED_DIR / f"{battery}.parquet"
        n_rows = len(pd.read_parquet(parquet_path, columns=["soh"]))
        stride = max(1, n_rows // TARGET_POINTS)
        result = predict_battery_lightgbm(
            lgbm_model, parquet_path, cfg["feature_columns"], stride=stride
        )
        out_name = f"{cfg['model']}_approach{cfg['approach']}_{battery}.parquet"
        out_path = PREDICTIONS_DIR / out_name
        result.to_parquet(out_path, index=False)
        logger.info("Wrote %s (%d points, stride=%d)", out_path, len(result), stride)


if __name__ == "__main__":
    main()
