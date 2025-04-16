import torch
import torch.nn as nn
from base_preprocess_node import BasePreprocessNode
from node_registry import register_node
from sockets import socketio

@register_node(name="AutoReshape", category="Preprocessing", tags={"run_early": True})
class AutoReshape(BasePreprocessNode):
    inputs = [{"name": "in", "type": "tensor"}]
    outputs = [{"name": "out", "type": "tensor"}]
    widgets = [
        {
            "type": "text",
            "name": "shape_str",
            "value": "-1, 64",
            "options": {"placeholder": "e.g. -1, 64"}
        }
    ]
    size = [210, 70]

    def __init__(self, shape_str="-1, 64"):
        super().__init__()
        self.shape_str = shape_str
        self.shape = self.parse_shape(shape_str)

    def parse_shape(self, s):
        try:
            parsed = tuple(int(x.strip()) if x.strip() != "-1" else -1 for x in s.split(","))
            print(f"🔢 Parsed reshape dims: {parsed}")
            self.shape = parsed

            # Update out_features if no -1
            if -1 not in parsed:
                self.out_features = parsed[-1] if len(parsed) > 1 else parsed[0]
                socketio.emit("property_update", {
                    "node_id": self.graph_node_id,
                    "property": "out_features",
                    "value": self.out_features
                })

            return parsed
        except Exception as e:
            print(f"⚠️ Invalid shape_str '{s}': {e}")
            socketio.emit("toast", {
                "message": f"⚠️ Invalid shape: {s}"
            })
            return (-1,)

    def set_input_shape(self, tensor):
        super().set_input_shape(tensor)

        # If shape_str is empty, suggest one (flatten to 2D)
        if not self.shape_str.strip():
            batch = tensor.shape[0]
            flat = int(torch.prod(torch.tensor(tensor.shape[1:])))
            suggested = f"-1, {flat}"
            self.shape_str = suggested
            self.shape = self.parse_shape(suggested)

            socketio.emit("property_update", {
                "node_id": self.graph_node_id,
                "property": "shape_str",
                "value": suggested
            })

            print(f"💡 AutoReshape suggested shape: {suggested}")

    def get_layer(self):
        # Use the parsed shape directly
        class ReshapeLayer(nn.Module):
            def __init__(self, shape):
                super().__init__()
                self.shape = shape

            def forward(self, x):
                return x.reshape(x.size(0), *self.shape)

        return ReshapeLayer(self.shape)
