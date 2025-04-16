import torch
import os
from sockets import socketio
from node_registry import register_node

@register_node(name="ExportModel", category="Exporters", tags={"run_after_train": True})
class ExportModelNode:
    inputs = [
        {"name": "model", "type": "model_out"},
        {"name": "sample_tensor", "type": "tensor"}
    ]
    widgets = [
        {
            "type": "combo",
            "name": "export_format",
            "value": "onnx",
            "options": {"values": ["onnx", "safetensors", "pt", "pth"]}
        },
        {
            "type": "text",
            "name": "prefix",
            "value": "model"
        }
    ]
    size = [240, 100]

    def __init__(self, export_format="onnx", prefix="model"):
        self.model = None
        self.sample_tensor = None
        self.export_format = export_format
        self.prefix = prefix
        self.export_dir = "saved_models/exports"

    def build(self):
        if self.model is None:
            raise Exception("⚠️ No model provided to ExportModelNode")
        os.makedirs(self.export_dir, exist_ok=True)

        filename = self._generate_unique_filename(self.prefix, self.export_format)
        path = os.path.join(self.export_dir, filename)

        try:
            if self.export_format == "onnx":
                if self.sample_tensor is None:
                    raise Exception("⚠️ ONNX export requires sample_tensor")
                torch.onnx.export(self.model, self.sample_tensor, path,
                                  input_names=["input"], output_names=["output"])
            elif self.export_format in ["pt", "pth"]:
                torch.save(self.model.state_dict(), path)
            elif self.export_format == "safetensors":
                try:
                    from safetensors.torch import save_file
                    save_file(self.model.state_dict(), path)
                except ImportError:
                    raise Exception("❌ Install safetensors: pip install safetensors")
            else:
                raise Exception(f"❌ Unsupported export format: {self.export_format}")

            socketio.emit("toast", {"message": f"📤 Exported model to {path}"})
            print(f"📤 Exported model to {path}")
        except Exception as e:
            socketio.emit("toast", {"message": f"❌ Export failed: {str(e)}"})
            raise

        return path

    def _generate_unique_filename(self, prefix, ext):
        i = 0
        while True:
            name = f"{prefix}_{i}.{ext}" if i > 0 else f"{prefix}.{ext}"
            if not os.path.exists(os.path.join(self.export_dir, name)):
                return name
            i += 1
