import os
import torch
from sockets import socketio
from node_registry import register_node

@register_node(name="Save Model", category="Exporters", tags={"run_after_train": True})
class SaveModelNode:
    inputs = [
        {"name": "model", "type": "model_out"}
    ]
    outputs = []  # no tensor output, just utility
    widgets = [
        {"type": "text", "name": "prefix", "value": "model"}
    ]
    size = [220, 80]

    def __init__(self, prefix="model"):
        self.model = None
        self.prefix = prefix
        self.graph_node_id = None
        self.save_dir = "saved_models/pt"

    def build(self):
        if self.graph_node_id:
            socketio.emit("node_active", {"node_id": self.graph_node_id})

        if self.model is None:
            raise Exception("⚠️ No model provided to SaveModelNode")

        os.makedirs(self.save_dir, exist_ok=True)
        filename = self._generate_unique_filename(self.prefix, "pt")
        path = os.path.join(self.save_dir, filename)

        torch.save(self.model.state_dict(), path)
        print(f"💾 Model saved to {path}")

        if self.graph_node_id:
            socketio.emit("toast", {"message": f"✅ Model saved to {path}"})
            socketio.emit("node_inactive", {"node_id": self.graph_node_id})

        return path

    def _generate_unique_filename(self, prefix, ext):
        i = 0
        while True:
            name = f"{prefix}_{i}.{ext}" if i > 0 else f"{prefix}.{ext}"
            if not os.path.exists(os.path.join(self.save_dir, name)):
                return name
            i += 1
