
# 🟩 PCB_Defect_Detect

This repository contains custom PCB-Defect_Detection model powered by YOLOv8. And also have real-time **Printed Circuit Board (PCB) defect detection** system which enables users to upload PCB images and automatically identifies defects using a trained object detection model, with deployment using **Streamlit** for a simple web interface.

## 📌 Project Overview

This project leverages **YOLOv8** to detect manufacturing defects in PCBs. It includes training on a custom dataset, followed by deploying the model using **Streamlit** to allow users to test the system with their own images.

---

## 📁 Repository Structure

```
📦 PCB_Defect_Detect/
├── test_images/            # Sample test images for inference
├── test_predictions/       # Folder with saved prediction outputs
├── trained_model.pt        # YOLOv8 trained model
├── snippets/               # some sniipets showing actual testing
├── fault-detect-yolov8.ipynb  # Notebook used for training and evaluation
├── streamlit_deploy.py     # Streamlit web app for real-time inference
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🧪 Dataset

- **Dataset used**: [`pcb-defcts`](https://www.kaggle.com/datasets/akhatova/pcb-defects) on Kaggle
- **Format**: YOLO format annotations
- **Training environment**: Kaggle Notebook using **dual T4 GPUs**

---

## 🚀 Streamlit Deployment

Use the Streamlit app to run real-time detection on custom images:

```bash
streamlit run streamlit_deploy.py
```

The app allows users to upload `.jpg`, `.jpeg`, or `.png` images and displays the predicted bounding boxes for defects.

### Example Output:
![Demo](/snippets/demo.png)  <!-- Replace with actual image path if needed -->

---

## 🧠 Model Training (YOLOv8)

- Framework: [Ultralytics YOLOv8](https://docs.ultralytics.com)
- Training Steps:
  - Initially tested on Google Colab (CPU)
  - Switched to Kaggle for faster training using **T4 GPUs**
  - Custom training with `pcb-defcts` dataset

```python
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")  # or yolov8s.yaml for a smaller model
model.train(data="pcb.yaml", epochs=180)
```

---

## 🔧 Requirements

Install required packages:

```bash
pip install -r requirements.txt
```

Sample `requirements.txt`:
```
ultralytics
opencv-python
numpy
streamlit
Pillow
```

---

## ✍️ Author

**Sulojan Rajkumar**  
Third-Year Undergraduate, Department of Electronics and Telecommunication Engineering  
University of Moratuwa

---

## 📌 Acknowledgments

- Ultralytics for the YOLOv8 framework  
- Kaggle for free GPU training resources  
- The creators of the `pcb-defcts` dataset

---

## 📃 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
