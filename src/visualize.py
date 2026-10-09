"""Dataset exploration: per-class counts and ground-truth boxes drawn on an image.

    python -m src.visualize --config configs/data.yaml --image 01_short_07.jpg
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from src.config import load_config
from src.data import index_images, load_annotations


def print_summary(annotations: pd.DataFrame) -> None:
    per_class = annotations.groupby("name").agg(
        images=("file_name", "nunique"),
        boxes=("file_name", "size"),
    )
    box_w = annotations["xmax"] - annotations["xmin"]
    box_h = annotations["ymax"] - annotations["ymin"]
    print(per_class.to_string())
    print(f"\nBox size (px, original resolution): median {box_w.median():.0f} x {box_h.median():.0f}")


def draw_annotations(image_path: Path, annotations: pd.DataFrame, out_path: Path | None = None) -> None:
    raw_image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
    image = raw_image.copy()
    for row in annotations.itertuples():
        cv2.rectangle(image, (row.xmin, row.ymin), (row.xmax, row.ymax), (255, 255, 255), 2)
        cv2.putText(image, row.name, (row.xmin, row.ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    fig, ax = plt.subplots(2, 1, figsize=(15, 15))
    ax[0].imshow(raw_image)
    ax[0].set_title("Raw image")
    ax[1].imshow(image)
    ax[1].set_title("Annotated image")
    for a in ax:
        a.axis("off")
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--image", help="file name of an image to draw, e.g. 01_short_07.jpg")
    parser.add_argument("--save", type=Path, help="save the figure instead of showing it")
    args = parser.parse_args()

    raw_dir = Path(load_config(args.config)["raw_dir"])
    annotations = load_annotations(raw_dir / "Annotations")
    print_summary(annotations)

    if args.image:
        image_path = index_images(raw_dir / "images")[args.image]
        draw_annotations(image_path, annotations[annotations["file_name"] == args.image], args.save)


if __name__ == "__main__":
    main()
