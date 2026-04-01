from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")

    model.train(
        data=r"C:\Users\Adithya KL\Desktop\CAIS\Drone Detection Using CNN\Data\merged\data.yaml",
        epochs=50,
        imgsz=512,
        batch=8,
        workers=2,
        device=0,
        project="runs",
        name="yolov8s_drone",
        patience=10,
        plots=True,
        exist_ok=True,
    )

    print("YOLOv8s training complete.")