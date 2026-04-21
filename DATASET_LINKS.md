# 🔗 YOLO Models - Dataset Links

All datasets used for training the YOLO models. Download directly from Roboflow.

---

## Part A: Object Detection - Pothole Detection

**Dataset**: Pothole Detection v6  
**Source**: Roboflow Universe  
**Link**: https://universe.roboflow.com/georgia-institute-of-technology-bwqcc/pothole-detection-mev9q

**Details**:
- **Total Images**: 2,742 annotated images
- **Format**: YOLOv8 Object Detection (bounding boxes)
- **Annotations**: Pothole locations with bounding boxes
- **Use Case**: Real-world road inspection and maintenance

**Quick Download**:
```bash
pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("georgia-institute-of-technology-bwqcc").project("pothole-detection-mev9q")
dataset = project.versions(6).download("yolov8")
```

---

## Part B: Image Classification - Plant Disease

**Dataset**: Plant Disease Classification v1  
**Source**: Roboflow Universe  
**Link**: https://universe.roboflow.com/classification/plant-disease-classification-6wbxf

**Details**:
- **Total Images**: 434 annotated images
- **Classes**: 5 (curl, healthy, leaf, whitefly, yellowish)
- **Format**: Multi-Class Classification
- **Use Case**: Agricultural disease diagnosis

**Class Labels**:
- curl - Leaf curl disease
- healthy - Healthy plant
- leaf - Leaf damage
- whitefly - Whitefly infestation
- yellowish - Yellowing/nutrient deficiency

**Quick Download**:
```bash
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("plant-disease-classification-6wbxf")
dataset = project.versions(1).download("yolov8")
```

---

## Part C: Pose Estimation - Human Pose

**Dataset**: Human Pose v2  
**Source**: Roboflow Universe  
**Link**: https://universe.roboflow.com/-zzbsz/human-pose

**Details**:
- **Total Images**: 9,532 annotated images
- **Format**: YOLOv8 Pose Estimation (17 COCO keypoints)
- **Keypoints**: 17 COCO-format joint annotations
- **Use Case**: Human pose tracking, activity recognition

**COCO Keypoints** (17 total):
1. Nose
2. Left Eye
3. Right Eye
4. Left Ear
5. Right Ear
6. Left Shoulder
7. Right Shoulder
8. Left Elbow
9. Right Elbow
10. Left Wrist
11. Right Wrist
12. Left Hip
13. Right Hip
14. Left Knee
15. Right Knee
16. Left Ankle
17. Right Ankle

**Quick Download**:
```bash
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("human-pose")
dataset = project.versions(2).download("yolov8")
```

---

## Part D: Oriented Bounding Boxes (OBB) - OBB v2022

**Dataset**: OBB v2022  
**Source**: Roboflow Universe  
**Link**: https://universe.roboflow.com/anna-university-dwfvc/obb-oqqt6

**Details**:
- **Format**: Oriented Bounding Box Detection
- **License**: CC BY 4.0
- **Use Case**: Aerial imagery, satellite images, rotated object detection

**Quick Download**:
```bash
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("anna-university-dwfvc").project("obb-oqqt6")
dataset = project.download("yolov8")
```

---

## Dataset Comparison Table

| Task | Name | Images | Format | Keypoints |
|------|------|--------|--------|-----------|
| Detection | Pothole Detection v6 | 2,742 | YOLO Detect | - |
| Classification | Plant Disease v1 | 434 | YOLO Classify | - |
| Pose | Human Pose v2 | 9,532 | YOLO Pose | 17 (COCO) |
| OBB | OBB v2022 | Variable | YOLO OBB | - |

---

## How to Use These Datasets

### Method 1: Download via Roboflow (Recommended)

1. Create account at: https://roboflow.com
2. Visit dataset link above
3. Click "Download" button
4. Select "YOLOv8" format
5. Choose download option (API or direct)

### Method 2: Use Roboflow API

```python
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="your_api_key_here")

# Download Detection dataset
project = rf.workspace("georgia-institute-of-technology-bwqcc").project("pothole-detection-mev9q")
dataset = project.versions(6).download("yolov8")

# Train with Ultralytics
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
results = model.train(data=f'{dataset.location}/data.yaml', epochs=100, imgsz=640)
```

### Method 3: Manual Download

1. Go to dataset link
2. Sign in with Roboflow account
3. Click "Export" → "Download ZIP"
4. Extract and organize by task

---

## Dataset Organization

After downloading, organize as:

```
datasets/
├── detection/
│   ├── images/
│   ├── labels/
│   └── data.yaml
├── classification/
│   ├── train/
│   ├── val/
│   └── data.yaml
├── pose/
│   ├── images/
│   ├── labels/
│   └── data.yaml
└── obb/
    ├── images/
    ├── labels/
    └── data.yaml
```

---

## License

- **Pothole Detection**: Public
- **Plant Disease**: Public
- **Human Pose**: Public
- **OBB v2022**: CC BY 4.0

Always cite the datasets when using them in publications.

---

## References

- [Roboflow Documentation](https://docs.roboflow.com/)
- [Ultralytics Dataset Guide](https://docs.ultralytics.com/datasets/)
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2016)

*Updated: April 21, 2026*
