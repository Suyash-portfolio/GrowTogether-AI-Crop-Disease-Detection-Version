# GrowTogether-AI: ML Pipeline Debugging Report

## 1. Root Cause Analysis
The "wrong predictions for all images" issue was caused by:
*   **Color Space Inversion:** The system was passing BGR images to models trained on RGB.
*   **Normalization Mismatch:** Missing ImageNet mean/std normalization for EfficientNet-B4.
*   **Model State:** Models were not explicitly set to `.eval()` mode, causing inconsistent behavior due to Dropout and BatchNorm.
*   **Coordinate Misalignment:** Potential issues in cropping logic where YOLO boxes didn't map correctly to the input image scale.

## 2. Debugging Checklist
- [x] **Verify Color Format:** Ensure `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` is called before PyTorch inference.
- [x] **Check Normalization:** Use `mean=[0.485, 0.456, 0.406]` and `std=[0.229, 0.224, 0.225]`.
- [x] **Set Eval Mode:** Always call `model.eval()` after loading weights.
- [x] **Validate Label Mapping:** Ensure the `labels` array indices match the training `class_to_idx` mapping.
- [x] **Crop Integrity:** Check if `img[y1:y2, x1:x2]` produces a valid, non-empty leaf image.

## 3. Accuracy Optimization (Target: 90%+)
To improve accuracy beyond the current baseline:
1.  **Tiling Strategy:** For high-res drone images (4K), split the image into 1024x1024 tiles with 20% overlap. Run YOLO on tiles rather than the full image to detect small disease spots.
2.  **Test-Time Augmentation (TTA):** Run inference on the original crop plus 3 rotated versions (90°, 180°, 270°) and average the probabilities.
3.  **Class Weights:** If your dataset is imbalanced (e.g., more "Healthy" than "Blight"), use `WeightedRandomSampler` during training.
4.  **Fine-Tuning:** Unfreeze the last 2 blocks of EfficientNet-B4 and retrain with a very low learning rate (1e-5) on your specific drone dataset.

## 5. Retraining Strategy (Multi-Crop Support)
To fix the "Mango/Banana misclassification" issue, the system is transitioning to a multi-crop model:
1.  **Dataset:** Using the full **PlantVillage Dataset** (38 classes) via `kagglehub`.
2.  **Download:** Run `python training/download_dataset.py` to fetch the latest multi-crop data.
3.  **Training:** Run `python training/train_classifier.py` to retrain the EfficientNet-B4 head on all 38 classes.
4.  **Validation:** The **Hybrid AI Validation Layer** (Gemini) remains active during this transition to catch any remaining Out-of-Distribution errors.
