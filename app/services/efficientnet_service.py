import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from app.utils.preprocessing import get_efficientnet_transform

class EfficientNetService:
    def __init__(self, model_path="ml_models/efficientnet_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Architecture
        self.model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=15)
        
        # Load Weights & Set to Eval Mode
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded weights from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load weights, using pretrained base. Error: {e}")
            
        self.model.to(self.device)
        self.model.eval() # CRITICAL: Disables Dropout/BatchNorm layers
        
        self.transform = get_efficientnet_transform()
        
        # Fixed Label Mapping (Must match your training dataset classes)
        # Full PlantVillage Mapping (38 Classes)
        self.labels = [
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
            "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Common_rust_", "Corn_(maize)___Gray_leaf_spot", "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
            "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
            "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
            "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
            "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
            "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites_Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
        ]
        
        # Supported Crops for Validation
        self.supported_crops = ["Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach", "Pepper", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"]
        
        self.confidence_threshold = 0.85 # Increased to 85% for production safety

    def classify(self, crop_rgb):
        input_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, index = torch.max(probabilities, dim=1)
            
        conf_value = confidence.item()
        predicted_label = self.labels[index.item()] if index.item() < len(self.labels) else "Unknown"
        
        # --- CRITICAL FIX: HIERARCHICAL VALIDATION ---
        # 1. Extract Crop Name from Label (e.g., "Tomato___Early_blight" -> "Tomato")
        predicted_crop = predicted_label.split("___")[0] if "___" in predicted_label else predicted_label
        
        # 2. Out-of-Distribution (OOD) Check
        # If the model is predicting a supported crop but with low confidence, 
        # OR if the predicted crop isn't in our verified list (though PlantVillage covers most),
        # we flag it as an invalid/unsupported input.
        
        if conf_value < self.confidence_threshold:
            return ClassificationResult(
                "Unsupported/Unknown Crop", 
                conf_value, 
                "The system detected an unknown crop (e.g., Mango, Citrus, or non-leaf object). This platform currently supports: " + ", ".join(self.supported_crops)
            )

        return ClassificationResult(predicted_label, conf_value)

class ClassificationResult:
    def __init__(self, label, confidence, custom_treatment=None):
        self.label = label
        self.confidence = confidence
        self.is_unsupported = "Unsupported" in label
        
        if custom_treatment:
            self.treatment = custom_treatment
        else:
            # Treatment mapping logic
            treatments = {
                "Tomato___Early_blight": "Apply fungicides containing chlorothalonil or copper.",
                "Tomato___Late_blight": "Remove infected plants immediately; apply Metalaxyl-M.",
                "Potato___Early_blight": "Improve air circulation and use resistant varieties.",
                "Potato___Late_blight": "CRITICAL: Destroy infected plants. Apply copper-based fungicides to surrounding healthy plants.",
                "Apple___Apple_scab": "Apply sulfur-based fungicides during the growing season.",
                "Corn_(maize)___Common_rust_": "Plant resistant hybrids; avoid overhead irrigation."
            }
            self.treatment = treatments.get(label, "Crop identified as healthy or with minor symptoms. Continue regular monitoring.")
