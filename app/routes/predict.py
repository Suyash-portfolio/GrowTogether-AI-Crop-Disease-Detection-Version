from fastapi import APIRouter, UploadFile, File
from app.services.yolo_service import YOLOService
from app.services.efficientnet_service import EfficientNetService
from app.utils.preprocessing import preprocess_drone_image
import time

router = APIRouter()

# Initialize Services with Fixed Logic
yolo_detector = YOLOService()
classifier = EfficientNetService()

@router.post("/predict")
async def predict_disease(file: UploadFile = File(...), lat: float = 0.0, lng: float = 0.0):
    start_time = time.time()
    
    try:
        # 1. Preprocessing (Fixes BGR/RGB and Normalization)
        image_bytes = await file.read()
        img_rgb, _ = preprocess_drone_image(image_bytes)
        
        # 2. YOLOv8 Detection (Localization)
        detections = yolo_detector.detect(img_rgb)
        
        if not detections:
            return {
                "status": "Healthy", 
                "results": [], 
                "geo": {"lat": lat, "lng": lng},
                "latency": time.time() - start_time
            }

        # 3. Cascaded Inference: EfficientNet Classification on Crops
        final_results = []
        for box in detections:
            # CRITICAL: Crop from the RGB image
            crop = img_rgb[box.y1:box.y2, box.x1:box.x2]
            
            # Skip invalid crops
            if crop.size == 0: continue
            
            # Classification
            class_result = classifier.classify(crop)
            
            final_results.append({
                "box": box.to_dict(),
                "disease": class_result.label,
                "confidence": class_result.confidence,
                "treatment": class_result.treatment
            })

        return {
            "status": "Infection Detected",
            "results": final_results,
            "geo": {"lat": lat, "lng": lng},
            "latency": time.time() - start_time
        }
        
    except Exception as e:
        return {"error": str(e), "status": "Pipeline Error"}
