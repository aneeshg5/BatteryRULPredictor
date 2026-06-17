"""Centralized configuration: paths, hyperparameters, and constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = PROCESSED_DIR / "predictions"

BATTERIES = ["RW9", "RW10", "RW11", "RW12"]
TRAIN_BATTERY = "RW9"
TEST_BATTERIES = ["RW10", "RW11", "RW12"]

BATCH_SIZE = 512
LEARNING_RATE = 1e-3
HIDDEN_LAYERS = [15, 10]
EPOCHS = 6
WINDOW_SIZE = 50

SOH_THRESHOLD = 0.80
SEED = 42
