# 🦺 PPE Control with YOLOv8

A real-time **Personal Protective Equipment (PPE) detection system** built with YOLOv8 and OpenCV. The system automatically detects whether personnel on a site are wearing required safety equipment — helmet, safety vest, mask — and flags violations in real time.

This project was originally developed as part of a research initiative (TÜBİTAK project) for automated workplace safety inspection.

## ✨ Features

- Real-time detection from video streams or static images
- Detects 10 classes: `Hardhat`, `NO-Hardhat`, `Mask`, `NO-Mask`, `Safety Vest`, `NO-Safety Vest`, `Person`, `Safety Cone`, `machinery`, `vehicle`
- Visual overlay of bounding boxes and safety status (safe/unsafe color coding)
- Object-oriented implementation (`PPEChecker` class) for easy integration into other pipelines
- Includes a full project report (PDF) documenting the methodology and results

## 🧰 Tech Stack

- Python
- [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics)
- OpenCV
- cvzone

## 📁 Project Structure

```
├── PPEChecker.py              # Core OOP class for PPE detection
├── TubitakProje.py            # Procedural implementation
├── TubitakProjeOOP.py         # OOP-based implementation
├── PPE Project Final Report.pdf
├── images/                    # Sample images
└── Videos/                    # Sample/test videos
```

## 🚀 Getting Started

```bash
pip install ultralytics opencv-python cvzone
```

```python
from PPEChecker import PPEChecker

checker = PPEChecker(model_path="best.pt", video_source="Videos/sample.mp4")
checker.run()
```

> **Note:** The trained YOLOv8 weights file is not included in this repository due to file size limits. Contact **yaltay556@gmail.com** to request access to the weights.

## 📄 Report

A detailed write-up of the project, dataset, and evaluation results is available in [`PPE Project Final Report.pdf`](./PPE%20Project%20Final%20Report.pdf).

## 📬 Contact

For questions or collaboration, reach out at **yaltay556@gmail.com**.
