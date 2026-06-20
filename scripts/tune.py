import argparse
import json
import logging

import torch

from battery_rul.config import PROCESSED_DIR, TRAIN_BATTERY, WINDOW_SIZE
from battery_rul.data.dataset import BatteryDataset
from battery_rul.training.tuning import run_study

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1, help="Window stride for subsampling")
    parser.add_argument("--device", default=None, help="Override the auto-selected device")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    train_dataset = BatteryDataset(
        PROCESSED_DIR / f"{TRAIN_BATTERY}.parquet", window_size=WINDOW_SIZE, stride=args.stride
    )
    device = torch.device(args.device) if args.device else None
    study = run_study(train_dataset, n_trials=args.n_trials, device=device)

    print("Best params:", study.best_params)
    print("Best value (val RMSE):", study.best_value)

    output_path = PROCESSED_DIR / "best_params.json"
    output_path.write_text(json.dumps(study.best_params, indent=2))
    logger.info("Saved best params to %s", output_path)


if __name__ == "__main__":
    main()
