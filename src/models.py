from pathlib import Path

from ultralytics import RTDETR, YOLO


def load_model(weights: str | Path) -> YOLO | RTDETR:
    """Load a YOLO or RT-DETR model from a model name (`yolov8s.pt`) or a trained checkpoint."""
    if "rtdetr" in Path(weights).stem.lower():
        return RTDETR(weights)
    model = YOLO(weights)
    # A fine-tuned RT-DETR checkpoint is usually just `best.pt`, so check the architecture too.
    if type(model.model).__name__ == "RTDETRDetectionModel":
        return RTDETR(weights)
    return model
