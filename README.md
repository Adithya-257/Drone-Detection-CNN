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
