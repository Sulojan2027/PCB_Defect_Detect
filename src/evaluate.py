"""Evaluate trained weights on a dataset split and save the metrics as JSON.

    python -m src.evaluate --weights runs/pcb/yolov8s/weights/best.pt --split test
"""

import argparse
import json
from pathlib import Path

from src.models import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data/processed/pcb.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/pcb_eval")
    parser.add_argument("--name", default=None, help="defaults to <run name>_<split>")
    args = parser.parse_args()

    weights = Path(args.weights)
    run_name = args.name or f"{weights.parent.parent.name}_{args.split}"

    model = load_model(weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(Path(args.project).resolve()),  # relative paths get nested under runs/detect/
        name=run_name,
    )

    summary = {
        "weights": str(weights),
        "split": args.split,
        **metrics.results_dict,
        "per_class_mAP50-95": {metrics.names[i]: float(metrics.box.maps[i]) for i in metrics.ap_class_index},
        "speed_ms_per_image": metrics.speed,
    }
    out = Path(metrics.save_dir) / "metrics.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Metrics written to {out}")


if __name__ == "__main__":
    main()
