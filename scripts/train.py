import argparse
import logging

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from battery_rul.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    PROCESSED_DIR,
    SEED,
    TEST_BATTERIES,
    TRAIN_BATTERY,
    WINDOW_SIZE,
)
from battery_rul.data.dataset import BatteryDataset
from battery_rul.data.preprocess import FEATURE_COLUMNS
from battery_rul.models.attention import BatteryAttention
from battery_rul.models.dnn import PaperDNN, UpgradedDNN
from battery_rul.models.lstm import BatteryLSTM
from battery_rul.training.trainer import Trainer, get_default_device
from battery_rul.training.tree_trainer import fit_lightgbm

logger = logging.getLogger(__name__)

PAPER_FEATURE_COLUMNS = FEATURE_COLUMNS[:5]
SEQUENCE_MODELS = {"lstm", "attention"}
SEQUENCE_MODEL_STRIDE = 5


def build_model(name: str, input_dim: int) -> nn.Module:
    if name == "paper_dnn":
        return PaperDNN(input_dim=input_dim)
    if name == "upgraded_dnn":
        return UpgradedDNN(input_dim=input_dim, layer_sizes=[64, 32])
    if name == "lstm":
        return BatteryLSTM(input_size=input_dim)
    if name == "attention":
        return BatteryAttention(input_dim=input_dim)
    raise ValueError(f"Unknown model: {name}")


def build_eval_sets(
    model_name: str, feature_columns: list[str], approach: int, stride: int = 1
) -> tuple[DataLoader, dict[str, DataLoader]]:
    if approach == 1:
        dataset = BatteryDataset(
            PROCESSED_DIR / f"{TRAIN_BATTERY}.parquet",
            window_size=WINDOW_SIZE,
            feature_columns=feature_columns,
            stride=stride,
        )
        n_train = int(0.8 * len(dataset))
        train_dl = DataLoader(Subset(dataset, range(n_train)), batch_size=BATCH_SIZE, shuffle=True)
        eval_sets = {
            TRAIN_BATTERY: DataLoader(
                Subset(dataset, range(n_train, len(dataset))), batch_size=BATCH_SIZE
            )
        }
        return train_dl, eval_sets

    train_ds = BatteryDataset(
        PROCESSED_DIR / f"{TRAIN_BATTERY}.parquet",
        window_size=WINDOW_SIZE,
        feature_columns=feature_columns,
        stride=stride,
    )
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_sets = {
        name: DataLoader(
            BatteryDataset(
                PROCESSED_DIR / f"{name}.parquet",
                window_size=WINDOW_SIZE,
                feature_columns=feature_columns,
                stride=stride,
            ),
            batch_size=BATCH_SIZE,
        )
        for name in TEST_BATTERIES
    }
    return train_dl, eval_sets


def build_lightgbm_data(
    approach: int, feature_columns: list[str]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    columns = [*feature_columns, "soh"]
    if approach == 1:
        df = pd.read_parquet(PROCESSED_DIR / f"{TRAIN_BATTERY}.parquet", columns=columns)
        n_train = int(0.8 * len(df))
        return df.iloc[:n_train], {TRAIN_BATTERY: df.iloc[n_train:]}

    train_df = pd.read_parquet(PROCESSED_DIR / f"{TRAIN_BATTERY}.parquet", columns=columns)
    eval_dfs = {
        name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet", columns=columns)
        for name in TEST_BATTERIES
    }
    return train_df, eval_dfs


def run_lightgbm(approach: int) -> None:
    train_df, eval_dfs = build_lightgbm_data(approach, FEATURE_COLUMNS)
    run_name = f"lightgbm_approach{approach}"
    results = fit_lightgbm(train_df, eval_dfs, FEATURE_COLUMNS, run_name)

    print(f"\nResults for lightgbm (Approach {approach}):")
    for name, rmse_value in results.items():
        print(f"  {name}: RMSE = {rmse_value * 100:.2f}%")
    print(f"  Average RMSE = {sum(results.values()) / len(results) * 100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["paper_dnn", "upgraded_dnn", "lstm", "attention", "lightgbm"],
    )
    parser.add_argument("--approach", required=True, type=int, choices=[1, 2])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default=None, help="Override the auto-selected device")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(SEED)

    if args.model == "lightgbm":
        run_lightgbm(args.approach)
        return

    feature_columns = PAPER_FEATURE_COLUMNS if args.model == "paper_dnn" else FEATURE_COLUMNS
    input_mode = "sequence" if args.model in SEQUENCE_MODELS else "flat"
    device = torch.device(args.device) if args.device else get_default_device()
    logger.info("Using device: %s", device)

    stride = SEQUENCE_MODEL_STRIDE if args.model in SEQUENCE_MODELS else 1
    train_dl, eval_sets = build_eval_sets(args.model, feature_columns, args.approach, stride=stride)
    model = build_model(args.model, input_dim=len(feature_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    run_name = f"{args.model}_approach{args.approach}"
    trainer = Trainer(model, optimizer, device, run_name, input_mode=input_mode)

    _, val_dl = next(iter(eval_sets.items()))
    trainer.fit(train_dl, val_dl, epochs=args.epochs)

    print(f"\nResults for {args.model} (Approach {args.approach}):")
    rmse_values = []
    for name, dl in eval_sets.items():
        metrics = trainer.eval_epoch(dl)
        rmse_values.append(metrics["rmse"])
        print(f"  {name}: RMSE = {metrics['rmse'] * 100:.2f}%")
    print(f"  Average RMSE = {sum(rmse_values) / len(rmse_values) * 100:.2f}%")


if __name__ == "__main__":
    main()
