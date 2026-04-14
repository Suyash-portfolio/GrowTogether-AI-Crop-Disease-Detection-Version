from ultralytics import YOLO
import torch

class YOLOService:
    def __init__(self, model_path="ml_models/yolov8_model.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load YOLOv8
        self.model = YOLO(model_path)
        print(f"YOLOv8 loaded on {self.device}")

    def detect(self, image_rgb):
        """
        Runs detection and returns bounding boxes.
        """
        # YOLOv8 handles RGB numpy arrays correctly
        results = self.model.predict(image_rgb, conf=0.25, device=self.device)
        
        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy() # Get boxes in [x1, y1, x2, y2]
            confs = r.boxes.conf.cpu().numpy()
            
            for i in range(len(boxes)):
                detections.append(DetectionBox(
                    x1=int(boxes[i][0]),
                    y1=int(boxes[i][1]),
                    x2=int(boxes[i][2]),
                    y2=int(boxes[i][3]),
                    conf=float(confs[i])
                ))
        return detections

class DetectionBox:
    def __init__(self, x1, y1, x2, y2, conf):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.conf = conf
    
    def to_dict(self):
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2, "conf": self.conf}
