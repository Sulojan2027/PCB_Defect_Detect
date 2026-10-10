---
license: mit
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - yolov8
  - object-detection
  - pcb
  - defect-detection
---

# PCB Defect Detection (YOLOv8s)

YOLOv8s fine-tuned to detect six PCB manufacturing defects: `missing_hole`, `mouse_bite`,
`open_circuit`, `short`, `spur` and `spurious_copper`.

## Usage

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model = YOLO(hf_hub_download("Sulojan/pcb-defect-yolov8s", "best.pt"))
results = model.predict("board.jpg", conf=0.3)
results[0].show()
```

## Training

- Data: Kaggle [akhatova/pcb-defects](https://www.kaggle.com/datasets/akhatova/pcb-defects)
  (693 images, 10 board designs), split 80/10/10 with seed 42
- Input 640×640, 180 epochs, AdamW (lr0 0.001), mixup 0.3; best checkpoint at epoch 116

## Results (held-out test set, 70 images, 286 boxes)

| Class | mAP50 | mAP50-95 |
|---|---|---|
| **all** | **0.941** | **0.503** |
| missing_hole | 0.995 | 0.657 |
| mouse_bite | 0.922 | 0.512 |
| open_circuit | 0.992 | 0.538 |
| short | 0.975 | 0.502 |
| spur | 0.904 | 0.448 |
| spurious_copper | 0.861 | 0.362 |

## Limitations

The model was trained on images of 10 board designs under controlled lighting. Confidence drops
on boards, cameras and lighting that differ from the training data.

Source code: <https://github.com/Sulojan2027/PCB_Defect_Detect>
