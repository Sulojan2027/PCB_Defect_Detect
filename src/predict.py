"""Run inference on an image or folder and save annotated predictions.

    python -m src.predict --weights runs/pcb/yolov8s/weights/best.pt --source test_images --out test_predictions
"""

import argparse
from pathlib import Path

import cv2

from src.models import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True, help="image file or folder")
    parser.add_argument("--out", default="test_predictions")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        stream=True,
        verbose=False,
    )
    for r in results:
        name = Path(r.path).name
        cv2.imwrite(str(out_dir / f"pred_{name}"), r.plot())
        counts = {model.names[int(c)]: int((r.boxes.cls == c).sum()) for c in r.boxes.cls.unique()}
        print(f"{name}: {counts or 'no defects'}")


if __name__ == "__main__":
    main()
