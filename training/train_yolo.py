from ultralytics import YOLO
import os

def train_yolo():
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Training parameters
    data_yaml = './training/data.yaml'
    epochs = 100
    imgsz = 640
    batch = 16

    # Start Training
    print(f"Starting YOLOv8 training on {data_yaml}...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='growtogether_yolo_v1',
        device=0, # Use GPU 0
        augment=True # Enable built-in mosaic and mixup augmentations
    )

    # Export the model to ONNX for Edge AI deployment
    print("Exporting trained model to ONNX for Edge deployment...")
    path = model.export(format='onnx')
    print(f"Trained model exported to {path}")

if __name__ == "__main__":
    train_yolo()
