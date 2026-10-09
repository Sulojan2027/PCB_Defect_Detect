"""Download the Kaggle PCB defects dataset and symlink it to `raw_dir` from the data config.

    python -m src.download_data --config configs/data.yaml

Uses `kagglehub`, which caches the download under ~/.cache/kagglehub. If Kaggle asks
for authentication, create an API token at https://www.kaggle.com/settings and either
save it as ~/.kaggle/kaggle.json or export KAGGLE_USERNAME and KAGGLE_KEY.
"""

import argparse
from pathlib import Path

import kagglehub

from src.config import load_config

KAGGLE_DATASET = "akhatova/pcb-defects"


def find_dataset_root(download_dir: Path) -> Path:
    """Locate the folder that contains both `images/` and `Annotations/` (PCB_DATASET)."""
    for candidate in [download_dir, *sorted(p for p in download_dir.rglob("*") if p.is_dir())]:
        if (candidate / "images").is_dir() and (candidate / "Annotations").is_dir():
            return candidate
    raise FileNotFoundError(f"No folder with images/ and Annotations/ found under {download_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    link = Path(load_config(args.config)["raw_dir"])

    download_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET, force_download=args.force))
    dataset_root = find_dataset_root(download_dir).resolve()
    print(f"Dataset available at {dataset_root}")

    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"{link} exists and is not a symlink; remove it or change raw_dir")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(dataset_root, target_is_directory=True)
    print(f"Linked {link} -> {dataset_root}")


if __name__ == "__main__":
    main()
