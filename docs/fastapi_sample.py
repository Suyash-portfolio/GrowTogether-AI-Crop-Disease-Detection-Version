from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from efficientnet_pytorch import EfficientNet

app = FastAPI()

# Load Models
yolo_model = YOLO("yolov8n-crop.pt")
classifier = EfficientNet.from_pretrained('efficientnet-b4', num_classes=15)
classifier.eval()

class DetectionResult(BaseModel):
    disease: str
    confidence: float
    lat: float
    lng: float
    treatment: str

@app.post("/api/v1/detect", response_model=DetectionResult)
async def detect_disease(lat: float, lng: float, file: UploadFile = File(...)):
    # 1. Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. YOLOv8 Localization
    results = yolo_model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) == 0:
        return {"disease": "Healthy", "confidence": 0.99, "lat": lat, "lng": lng, "treatment": "None"}

    # 3. EfficientNet Classification on first detected box
    x1, y1, x2, y2 = map(int, boxes[0])
    roi = img[y1:y2, x1:x2]
    roi = cv2.resize(roi, (380, 380)) # EfficientNet-B4 input size
    
    # Preprocess for PyTorch
    roi_tensor = torch.from_numpy(roi).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    
    with torch.no_grad():
        output = classifier(roi_tensor)
        prob = torch.softmax(output, dim=1)
        conf, idx = torch.max(prob, dim=1)

    # Mock mapping for demo
    disease_map = {0: "Potato Late Blight", 1: "Tomato Leaf Mold", 2: "Corn Rust"}
    treatment_map = {
        "Potato Late Blight": "Apply Fungicides (Metalaxyl)",
        "Tomato Leaf Mold": "Improve Ventilation",
        "Corn Rust": "Use Resistant Hybrids"
    }

    disease_name = disease_map.get(idx.item(), "Unknown Disease")

    return {
        "disease": disease_name,
        "confidence": float(conf.item()),
        "lat": lat,
        "lng": lng,
        "treatment": treatment_map.get(disease_name, "Consult Agronomist")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
