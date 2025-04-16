import torch
import torch.nn as nn
from base_model_node import BaseModelNode, mark_active
from node_registry import register_node

@register_node(name="Dense Layer", category="Model")
class DenseLayer(BaseModelNode):
    inputs = [
        {"name": "in", "type": "tensor"}
    ]
    outputs = [
        {"name": "out", "type": "tensor"}
    ]
    widgets = [
        {"type": "number", "name": "in_features", "value": 128, "options": {"min": 1, "max": 1024}},
        {"type": "number", "name": "out_features", "value": 64, "options": {"min": 1, "max": 1024}},
        {"type": "combo", "name": "activation", "value": "relu", "options": {"values": ["relu", "sigmoid", "none"]}},
        {"type": "toggle", "name": "lock_in_features", "value": False}
    ]
    size = [180, 100]

    def __init__(self, in_features=128, out_features=64, activation="relu", lock_in_features=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.lock_in_features = lock_in_features

    @mark_active
    def get_layer(self):
        if self.input_tensor_shape and not self.lock_in_features:
            self.in_features = int(torch.prod(torch.tensor(self.input_tensor_shape)))
            self.emit_update("in_features", self.in_features)
            print(f"🔁 DenseLayer auto-adjusted in_features = {self.in_features}")
        elif self.input_tensor_shape is None:
            print(f"ℹ️ Fallback in_features = {self.in_features}")

        self.emit_update("out_features", self.out_features)

        return nn.Sequential(
            nn.Linear(self.in_features, self.out_features),
            self.get_activation()
        )
