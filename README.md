# Drone-Detection-CNN

A real-time aerial object detection system fine-tuned on YOLOv8s, trained on a merged dataset of 12,700+ images across 4 classes. Deployed as a live interactive web app with image upload, video processing, and preloaded demo examples.

## 🚀 Live Demo

https://drone-detection-cnn.streamlit.app

## Screenshots

App Interface
![preview](https://github.com/user-attachments/assets/3ce19a40-562a-48ee-b726-3153749570c7)

Multi-class Detection — Drone & Plane
![multi class detection-Drone   Plane](https://github.com/user-attachments/assets/f8dd0bc8-7cea-485e-b88b-96dce689d97e)

Video Processing — Bird vs Drone
![Video Processing- Bird VS Drone](https://github.com/user-attachments/assets/e347384f-1ffc-4dc3-866a-c74ce9ba20cb)

Demo Examples
![demo examples](https://github.com/user-attachments/assets/8d952bc5-ad46-474c-aed5-534a5c465e30)

## Features

- Image upload — upload any aerial image and get bounding box detections with confidence scores

- Video processing — upload or use the preloaded Bird vs Drone real-world footage; model processes every frame and returns an annotated preview + downloadable MP4
 
- Demo examples — 9 preloaded images covering drones, birds, planes and multi-class scenes
  
- Adjustable confidence threshold — tune the detection sensitivity via a sidebar slider
  
- 4-class detection — drone, bird, plane, kite with colour-coded bounding boxes


## Model Comparison

| Metric             | YOLOv8n  | YOLOv8s  |
|:-------------------|:---------:|:---------:|
| **mAP@50 (overall)** | 0.764     | **0.820** |
| mAP@50 (drone)     | **0.875** | 0.873     |
| mAP@50 (bird)      | 0.711     | **0.713** |
| mAP@50 (plane)     | 0.914     | **0.932** |
| mAP@50 (kite)      | 0.559     | **0.761** |
| **Precision**      | 0.552     | **0.742** |
| **Recall**         | 0.762     | **0.802** |
| Inference Speed    | **~2ms**  | ~3ms      |
| Model Size         | **6.2MB** | 22.5MB    |

YOLOv8s was selected for deployment due to higher overall mAP50 and significantly better precision. YOLOv8n remains the better choice for edge deployment scenarios where speed and model size are constrained.

## TRAINING CURVES

<table>
  <tr>
    <td align="center"><b>YOLOv8n ⚡</b></td>
    <td align="center"><b>YOLOv8s 🚀</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/a7ae6fc8-a4d9-4064-ad1e-777ff1a44ae4" width="100%">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/6bb606fc-4ad5-4fe7-ad62-cc6bf5a23ceb" width="100%">
    </td>
  </tr>
</table>


### 1) Confusion Matrix

<table>
  <tr>
    <td align="center"><b>YOLOv8n ⚡</b></td>
    <td align="center"><b>YOLOv8s 🚀</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/f67b1634-4dd4-405a-8778-c3f591da7629" width="100%">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/171aad88-9b67-4893-b513-95cede9c6734" width="100%">
    </td>
  </tr>
</table>

### 2) Precision-Recall Curve 

<table>
  <tr>
    <td align="center"><b>YOLOv8n ⚡</b></td>
    <td align="center"><b>YOLOv8s 🚀</b></td>
  </tr>
  <tr>
    <td>
      <img src="https://github.com/user-attachments/assets/20462efc-fc52-4d22-9247-c1af9bdb20bf" width="100%">
    </td>
    <td>
      <img src="https://github.com/user-attachments/assets/03256f6c-db6c-428a-949f-0d7d5c23fd81" width="100%">
    </td>
  </tr>
</table>

## Validation Predictions- YOLOv8s

![val_batch2_pred](https://github.com/user-attachments/assets/124715a9-16cc-4ac2-b104-940a9de5ca43)

## Datasets
Three Roboflow datasets were merged into a unified training set:

| Dataset              | Images  | Classes                              |
|:--------------------|:-------:|:-------------------------------------|
| Drone Dataset (7k)  | ~6,996  | drone, not-drone                     |
| DroneorBird (5k)    | ~5,079  | bird, drone, helicopter, plane       |
| Drone or Kite (600) | ~651    | bird, drone, kite, plane             |

**Total: 12,726 images across train/val/test (80/10/10 split)**

Key data engineering decisions:

- Polygon annotations in the 7k dataset were converted to standard YOLO bounding box format
- Inconsistent class labels across datasets were remapped to a unified 4-class scheme (drone, bird, plane, kite)
- Annotation quality was audited — labelling inconsistencies were identified and handled
- Classes were balanced to ensure the model learns to distinguish drones from visually similar objects (birds, kites)

## Training
```
Model:      YOLOv8s (fine-tuned from pretrained COCO weights)
Epochs:     50
Image size: 512px
Batch size: 8
Device:     NVIDIA RTX 3050 (4GB VRAM)
Framework:  Ultralytics
```

