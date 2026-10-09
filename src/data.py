"""Reading the raw Kaggle PCB dataset (Pascal VOC XML annotations)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_xml(xml_file: str | Path) -> list[dict]:
    root = ET.parse(xml_file).getroot()
    file_name = root.find("filename").text
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    data = []
    for obj in root.findall("object"):
        data.append({
            "file_name": file_name,
            "width": width,
            "height": height,
            "name": obj.find("name").text.strip().lower(),
            "xmin": int(obj.find("bndbox/xmin").text),
            "ymin": int(obj.find("bndbox/ymin").text),
            "xmax": int(obj.find("bndbox/xmax").text),
            "ymax": int(obj.find("bndbox/ymax").text),
        })
    return data


def load_annotations(annot_dir: str | Path) -> pd.DataFrame:
    """One row per bounding box across every XML file under `annot_dir`."""
    rows = []
    for xml_path in sorted(Path(annot_dir).rglob("*.xml")):
        rows.extend(parse_xml(xml_path))
    return pd.DataFrame(rows)


def index_images(images_dir: str | Path) -> dict[str, Path]:
    """Map image file name -> path, searching all defect sub-folders."""
    return {
        p.name: p
        for p in Path(images_dir).rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES
    }


def to_yolo_boxes(annotations: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Convert pixel VOC boxes to normalized YOLO `class xc yc w h`.

    Normalized coordinates are independent of any later (stretch) resize, so the
    boxes are computed from the original resolution without rounding.
    """
    a = annotations
    return pd.DataFrame({
        "file_name": a["file_name"],
        "class_index": a["name"].map(classes.index),
        "x_center": ((a["xmin"] + a["xmax"]) / (2 * a["width"])).clip(0, 1),
        "y_center": ((a["ymin"] + a["ymax"]) / (2 * a["height"])).clip(0, 1),
        "width": ((a["xmax"] - a["xmin"]) / a["width"]).clip(0, 1),
        "height": ((a["ymax"] - a["ymin"]) / a["height"]).clip(0, 1),
    })
