"""PyTorch Dataset and DataLoader construction for windowed battery feature sequences."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from battery_rul.config import (
    BATCH_SIZE,
    PROCESSED_DIR,
    TEST_BATTERIES,
    TRAIN_BATTERY,
    WINDOW_SIZE,
)
from battery_rul.data.preprocess import FEATURE_COLUMNS


class BatteryDataset(Dataset):
    """Windowed (feature history, SOH target) pairs from one battery's processed parquet."""

    def __init__(self, parquet_path: Path, window_size: int = WINDOW_SIZE) -> None:
        df = pd.read_parquet(parquet_path, columns=[*FEATURE_COLUMNS, "soh"])
        self.features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        self.targets = df["soh"].to_numpy(dtype=np.float32)
        self.window_size = window_size

    def __len__(self) -> int:
        return len(self.targets) - self.window_size + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.features[idx : idx + self.window_size]
        target = self.targets[idx + self.window_size - 1]
        return torch.from_numpy(window), torch.tensor(target)


def get_dataloaders(
    train_battery: str = TRAIN_BATTERY,
    test_batteries: list[str] | None = None,
    batch_size: int = BATCH_SIZE,
    window_size: int = WINDOW_SIZE,
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Build the training DataLoader for train_battery and one per test battery."""
    test_batteries = test_batteries if test_batteries is not None else TEST_BATTERIES

    train_ds = BatteryDataset(processed_dir / f"{train_battery}.parquet", window_size)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_dls = {
        name: DataLoader(
            BatteryDataset(processed_dir / f"{name}.parquet", window_size),
            batch_size=batch_size,
            shuffle=False,
        )
        for name in test_batteries
    }
    return train_dl, test_dls
