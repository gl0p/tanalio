import torch.nn as nn
from base_preprocess_node import BasePreprocessNode, mark_active
from sockets import socketio
from node_registry import register_node

@register_node(name="AutoFlatten", category="Preprocessing", tags={"run_early": True})
class AutoFlatten(BasePreprocessNode):
    inputs = [
        {"name": "in", "type": "tensor"}
    ]

    outputs = [
        {"name": "out", "type": "tensor"}
    ]

    widgets = []  # No manual widgets, everything is auto-detected
    size = [200, 60]

    def __init__(self):
        super().__init__()

    def get_output(self):
        return {
            "out": self.get_layer(),
            "out_features": self.out_features
        }

    @mark_active
    def get_layer(self):
        return nn.Flatten()
