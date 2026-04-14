# GrowTogether-AI: Production System Architecture

## 1. Cascaded ML Pipeline
The system satisfies the **Detection Accuracy (25%)** and **Localization Precision (20%)** criteria through a three-stage cascaded architecture:

1.  **CNN Feature Extractor (`app/services/cnn_feature_extractor.py`):**
    *   Uses a ResNet-50 backbone to extract high-dimensional feature vectors.
    *   Identifies low-level anomalies (color shifts, texture irregularities) that signal potential disease.
2.  **YOLOv8 Localization (`app/services/yolo_service.py`):**
    *   Trained specifically on leaf-level datasets (PlantVillage + custom drone data).
    *   Outputs precise bounding boxes (ROIs) for infected regions, ensuring high localization precision.
3.  **EfficientNet-B4 Classification (`app/services/efficientnet_service.py`):**
    *   The state-of-the-art classifier for fine-grained disease identification.
    *   Analyzes the YOLO-cropped ROIs to provide the final disease label and treatment recommendation.

## 2. Hybrid AI Validation Layer
To address the limitations of the local model (specialized on Potato/Tomato) and prevent misclassifications of Out-of-Distribution (OOD) crops like Mango, the system employs a **Hybrid Validation Layer**:

*   **Local Inference (Primary):** CNN + YOLOv8 + EfficientNet-B4 handles real-time detection and classification of supported crops.
*   **Cloud Expert Analysis (Secondary):** Integrated with the **Gemini 3.0 Flash API**. When a scan is performed, the image and local prediction are sent to Gemini for "Expert Opinion".
*   **OOD Detection:** Gemini acts as a safety net to identify if the input is an unsupported crop (e.g., Mango, Citrus) and provides corrective scientific analysis.
*   **Transparency:** All external AI contributions are clearly disclosed in the UI as "Supporting Systems".

## 3. Multi-Crop Expansion Strategy
The system is designed for rapid scaling to a comprehensive multi-crop dataset:
*   **Dataset Diversity:** Future training cycles will incorporate the full 38-class PlantVillage dataset plus regional datasets for tropical crops (Mango, Banana, Coconut).
*   **Hierarchical Classification:** Implementing a "Crop First" classifier that identifies the plant species before attempting disease diagnosis.
*   **Transfer Learning:** Leveraging weights from the current specialized models to bootstrap training on new crop types.

## 4. Real-Time Processing & Edge AI (20%)
*   **Edge Optimization:** Models are exported to **ONNX** and **TorchScript** (`app/services/edge_inference.py`) for deployment on drone-side hardware (NVIDIA Jetson).
*   **Latency:** The pipeline is optimized for <1.5s inference time per high-res frame.
*   **Redis Caching:** Backend results are cached in Redis to prevent redundant processing of overlapping drone tiles.

## 3. GIS & PostGIS Integration (15%)
*   **Spatial Mapping:** Every detection is stored with `geometry(Point, 4326)` in PostgreSQL using **PostGIS**.
*   **Heatmaps:** The frontend uses Leaflet.js to render circular heatmaps, visualizing infection density and spread patterns across the field.

## 4. Alert System & Outbreak Prediction (20%)
*   **Farmer Notifications:** Integrated with Twilio for automated WhatsApp/SMS alerts.
*   **Severity-Based Alerts:** High-severity detections trigger immediate "Protocol Red" alerts with specific chemical treatment suggestions.

## 5. Model Training Pipeline
To achieve high accuracy and tailor the system to local variations, a comprehensive training module is included:

*   **Data Augmentation (`training/dataset.py`):** Implements agricultural-specific transforms (RandomVerticalFlip, ColorJitter, RandomRotation) to simulate drone flight conditions and lighting variations.
*   **Classifier Training (`training/train_classifier.py`):** Fine-tunes EfficientNet-B4 using differential learning rates (lower for backbone, higher for head) to preserve pretrained features while adapting to specific crop diseases.
*   **Detection Training (`training/train_yolo.py`):** Leverages the Ultralytics API for YOLOv8 training with built-in mosaic and mixup augmentations.
*   **Feature Extraction (`training/train_cnn.py`):** Trains a ResNet-50 backbone to extract robust texture features from leaf images.

## 6. Deployment Strategy
*   **Dockerized Microservices:** The FastAPI backend and ML services are containerized for horizontal scaling.
*   **Hybrid Cloud:** Edge AI handles real-time localization on the drone, while the Cloud Backend handles heavy classification and long-term spatial analytics.
