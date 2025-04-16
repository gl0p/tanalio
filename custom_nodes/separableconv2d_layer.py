import torch.nn as nn
from base_model_node import BaseModelNode
from node_registry import register_node
from sockets import socketio

@register_node(name="SeparableConv2D", category="Model")
class SeparableConv2D(BaseModelNode):
    inputs = [{"name": "in", "type": "tensor"}]
    outputs = [{"name": "out", "type": "tensor"}]
    widgets = [
        {"type": "number", "name": "in_channels", "value": 3, "options": {"min": 1, "max": 512}},
        {"type": "number", "name": "out_channels", "value": 16, "options": {"min": 1, "max": 512}},
        {"type": "number", "name": "kernel_size", "value": 3, "options": {"min": 1, "max": 11}},
        {"type": "combo", "name": "activation", "value": "relu", "options": {"values": ["relu", "sigmoid", "none"]}},
    ]
    size = [200, 120]

    def __init__(self, in_channels=3, out_channels=16, kernel_size=3, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.activation = activation

    def on_input_connected(self, input_index, source_node):
        # Inherit shape and auto-set in_channels
        super().on_input_connected(input_index, source_node)
        if hasattr(source_node, "input_tensor_shape") and source_node.input_tensor_shape:
            self.in_channels = source_node.input_tensor_shape[0]
            socketio.emit("property_update", {
                "node_id": self.graph_node_id,
                "property": "in_channels",
                "value": self.in_channels
            })

    def get_layer(self):
        layers = [
            nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size,
                      padding=self.kernel_size // 2, groups=self.in_channels),  # depthwise
            nn.Conv2d(self.in_channels, self.out_channels, 1)  # pointwise
        ]
        if self.activation == "relu":
            layers.append(nn.ReLU())
        elif self.activation == "sigmoid":
            layers.append(nn.Sigmoid())
        return nn.Sequential(*layers)
