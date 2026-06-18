"""Export the upgraded_dnn / approach-2 model from MLflow into a standalone artifact
the inference API can load without depending on the MLflow tracking store at request time.

python scripts/export_model.py
"""

import json
import logging

import mlflow.pytorch
import pandas as pd
import torch
from mlflow.tracking import MlflowClient

from battery_rul.config import (
    INFERENCE_DEFAULTS_PATH,
    MODEL_PATH,
    PROCESSED_DIR,
    TRAIN_BATTERY,
)

logger = logging.getLogger(__name__)

RUN_NAME = "upgraded_dnn_approach2"


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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    client = MlflowClient()
    run_id = latest_run_id(client, RUN_NAME)
    logger.info("Exporting %s from run %s", RUN_NAME, run_id)

    model = mlflow.pytorch.load_model(f"runs:/{run_id}/model")
    torch.save(model, MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)

    train_df = pd.read_parquet(
        PROCESSED_DIR / f"{TRAIN_BATTERY}.parquet",
        columns=["absolute_time_raw", "cycle_count_raw"],
    )
    defaults = {
        "absolute_time": float(train_df["absolute_time_raw"].mean()),
        "cycle_count": float(train_df["cycle_count_raw"].mean()),
    }
    with INFERENCE_DEFAULTS_PATH.open("w") as f:
        json.dump(defaults, f, indent=2)
    logger.info("Saved inference defaults to %s: %s", INFERENCE_DEFAULTS_PATH, defaults)


if __name__ == "__main__":
    main()
