import torch.onnx
from app.services.yolo_service import YOLOService

def export_to_onnx(model, dummy_input, path):
    """
    Converts PyTorch models to ONNX for Edge AI deployment on drones.
    """
    torch.onnx.export(model, dummy_input, path, opset_version=11)
    print(f"Model exported to {path} for Edge Inference")

class EdgeInference:
    def __init__(self, onnx_path):
        # Use ONNX Runtime for lightweight inference
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path)

    def run(self, input_data):
        return self.session.run(None, {"input": input_data})
