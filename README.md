# Pothole Detector for Road Maintenance

An AI-powered computer vision system for automatic pothole detection using deep learning. This project leverages YOLOv8 and open road imagery datasets to identify and localize potholes in street-level images and videos. The goal is to support road maintenance teams with faster inspections, better prioritization of repairs, and safer transportation infrastructure.

Key features:
	•	Automatic pothole detection from images or dashcam/drone footage
	•	YOLOv8-based object detection with easy training & fine-tuning
	•	Supports open datasets (RDD, Pothole-600, Indian Pothole Dataset)
	•	Includes tools for dataset conversion, splitting, and annotation QA
	•	Export to ONNX, CoreML, TensorRT, TFLite for edge deployment
	•	Designed for scalable road maintenance workflows


# Pothole Detection Starter (YOLOv8)

This repo is a quick-start for training a **pothole detector** using **YOLOv8** (Ultralytics).  
It is dataset-agnostic but assumes (or converts to) **YOLO box format** with a single class: `pothole`.

## What’s inside
- `notebooks/train_yolov8_potholes.ipynb` — end-to-end training + evaluation + export.
- `src/voc_to_yolo.py` — convert **Pascal VOC** annotations (XML) to YOLO format.
- `src/split_dataset.py` — split images/labels into train/val.
- `configs/data.yaml` — YOLO data config (points to `data/` here).
- `requirements.txt` — Python deps for local training.
- `data/` — put your dataset here (see layout below).

## Dataset options (open data)
You can use any of these (download separately):
- **RDD (Road Damage Dataset)** — multiple countries, has potholes/cracks. (Often VOC-style XML or CSV; use converter if needed)
- **Pothole-600** — smaller, focused dataset (often VOC XML).  
- **Indian Pothole Detection Dataset (Kaggle)** — detection-ready or CSV annotations.

> Tip: If your dataset provides **Pascal VOC (XML)** annotations, use `src/voc_to_yolo.py` to convert to YOLO format.

## Expected YOLO format (single class)
```
data/
  images/
    train/*.jpg
    val/*.jpg
  labels/
    train/*.txt   # each line: class_id cx cy w h (normalized 0..1)
    val/*.txt
```
- Class list (in `configs/data.yaml`): `[pothole]`  
- One `.txt` per image, same basename.

## Quickstart
1) Create & activate a new Python env (3.10+ recommended).
2) Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3) Prepare dataset:
   - If you have VOC XML: run `python src/voc_to_yolo.py --images <IMG_DIR> --ann <XML_DIR> --out data --split 0.85`
   - Or place your YOLO-ready images/labels under `data/` as shown.
4) Open the notebook:
   ```bash
   jupyter lab notebooks/train_yolov8_potholes.ipynb
   ```
5) Train in the notebook, or use the Python script snippet there.

## Exports & deployment
The notebook shows how to export to **ONNX**, **TensorRT**, **CoreML**, or **TFLite**.  
You can then run edge inference (Android/iOS/Jetson) at real-time-ish FPS.

## Notes
- If your dataset has **multiple classes** (e.g., cracks), edit `configs/data.yaml` accordingly.
- For **severity**, see the optional **MiDaS** depth cell in the notebook to score potholes by area+depth.
- Remember to **blur** faces/plates in your training data if present.
