"""Train a detector from a config file.

    python -m src.train --config configs/train/yolov8s.yaml
    python -m src.train --config configs/train/yolov8s.yaml epochs=1 device=cpu
"""

import argparse
from pathlib import Path

from src.config import load_config, parse_overrides
from src.models import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="key=value overrides for the `train:` section")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_args = {**cfg["train"], **parse_overrides(args.overrides)}
    # Ultralytics nests relative project paths under runs/detect/, so make it absolute.
    if "project" in train_args:
        train_args["project"] = str(Path(train_args["project"]).resolve())

    model = load_model(cfg["model"])
    model.train(**train_args)
    print(f"Best weights: {model.trainer.best}")


if __name__ == "__main__":
    main()
