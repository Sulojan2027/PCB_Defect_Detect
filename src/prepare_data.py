"""Convert the raw VOC dataset into a YOLO dataset with train/val/test splits.

    python -m src.prepare_data --config configs/data.yaml
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import yaml

from src.config import load_config
from src.data import index_images, load_annotations, to_yolo_boxes

SPLITS = ("train", "val", "test")


def split_files(file_names: list[str], ratios: dict, seed: int) -> dict[str, list[str]]:
    file_names = sorted(file_names)
    random.Random(seed).shuffle(file_names)

    n = len(file_names)
    n_train = int(n * ratios["train"])
    n_val = int(n * ratios["val"])
    return {
        "train": file_names[:n_train],
        "val": file_names[n_train:n_train + n_val],
        "test": file_names[n_train + n_val:] if ratios.get("test", 0) > 0 else [],
    }


def write_image(src: Path, dst: Path, resize: list[int] | None) -> None:
    if resize is None:
        shutil.copy(src, dst)
        return
    image = cv2.imread(str(src))
    cv2.imwrite(str(dst), cv2.resize(image, tuple(resize)))


def prepare(cfg: dict) -> Path:
    raw_dir = Path(cfg["raw_dir"])
    output_dir = Path(cfg["output_dir"])
    classes = cfg["classes"]

    annotations = load_annotations(raw_dir / "Annotations")
    unknown = set(annotations["name"]) - set(classes)
    if unknown:
        raise ValueError(f"Annotations contain classes not in config: {sorted(unknown)}")

    boxes = to_yolo_boxes(annotations, classes)
    images = index_images(raw_dir / "images")
    missing = set(boxes["file_name"]) - set(images)
    if missing:
        print(f"Warning: {len(missing)} annotated images not found, skipping them")
        boxes = boxes[~boxes["file_name"].isin(missing)]

    splits = split_files(boxes["file_name"].unique().tolist(), cfg["split"], cfg["seed"])

    # Start clean so a re-run with another seed can't leave stale files behind.
    for sub in ("images", "labels"):
        shutil.rmtree(output_dir / sub, ignore_errors=True)

    by_image = dict(tuple(boxes.groupby("file_name")))
    for split, file_names in splits.items():
        if not file_names:
            continue
        (output_dir / "images" / split).mkdir(parents=True)
        (output_dir / "labels" / split).mkdir(parents=True)
        for file_name in file_names:
            write_image(images[file_name], output_dir / "images" / split / file_name, cfg.get("resize"))
            label_path = output_dir / "labels" / split / f"{Path(file_name).stem}.txt"
            rows = by_image[file_name].itertuples(index=False)
            label_path.write_text("".join(
                f"{r.class_index} {r.x_center:.6f} {r.y_center:.6f} {r.width:.6f} {r.height:.6f}\n"
                for r in rows
            ))
        print(f"{split:<5}: {len(file_names)} images")

    dataset_yaml = output_dir / "pcb.yaml"
    dataset_yaml.write_text(yaml.safe_dump({
        "path": str(output_dir.resolve()),
        **{split: f"images/{split}" for split in SPLITS if splits[split]},
        "names": dict(enumerate(classes)),
    }, sort_keys=False))
    print(f"Dataset YAML written to {dataset_yaml}")
    return dataset_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    prepare(load_config(args.config))


if __name__ == "__main__":
    main()
