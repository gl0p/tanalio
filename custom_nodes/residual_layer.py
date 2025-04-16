import torch
import torch.nn as nn
from base_model_node import BaseModelNode
from node_registry import register_node
from sockets import socketio

@register_node(name="Residual Block", category="Model")
class ResidualBlock(BaseModelNode):
    inputs = [{"name": "in", "type": "tensor"}]
    outputs = [{"name": "out", "type": "tensor"}]
    widgets = [
        {"type": "number", "name": "channels", "value": 64, "options": {"min": 1, "max": 512}},
        {"type": "number", "name": "kernel_size", "value": 3, "options": {"min": 1, "max": 7}},
        {"type": "number", "name": "stride", "value": 1, "options": {"min": 1, "max": 3}},
        {"type": "number", "name": "padding", "value": 1, "options": {"min": 0, "max": 3}},
        {"type": "combo", "name": "activation", "value": "relu", "options": {"values": ["relu", "sigmoid", "none"]}}
    ]
    size = [220, 140]

    def __init__(self, channels=64, kernel_size=3, stride=1, padding=1, activation="relu"):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

    def set_input_shape(self, tensor):
        super().set_input_shape(tensor)
        # Optional: if stride != 1, adjust H/W
        if self.input_tensor_shape and len(self.input_tensor_shape) >= 3:
            C, H, W = self.input_tensor_shape[-3:]
            if C != self.channels:
                socketio.emit("toast", {
                    "message": f"⚠️ ResidualBlock expects input with {self.channels} channels, but got {C}."
                })

            H_out, W_out = self.compute_conv2d_output_shape(H, W, self.kernel_size, self.stride, self.padding)
            new_shape = [self.channels, H_out, W_out]
            self.input_tensor_shape = new_shape
            self.out_features = int(torch.prod(torch.tensor(new_shape)))

            socketio.emit("property_update", {
                "node_id": self.graph_node_id,
                "property": "out_features",
                "value": self.out_features
            })
            print(f"📐 ResidualBlock updated shape → {new_shape} → out_features = {self.out_features}")

    def get_layer(self):
        class ResidualModule(nn.Module):
            def __init__(self, channels, kernel_size, stride, padding, activation):
                super().__init__()
                act = {
                    "relu": nn.ReLU(),
                    "sigmoid": nn.Sigmoid(),
                    "none": nn.Identity()
                }.get(activation, nn.ReLU())

                self.block = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, stride=stride, padding=padding),
                    act,
                    nn.Conv2d(channels, channels, kernel_size, stride=1, padding=padding)
                )

            def forward(self, x):
                return self.block(x) + x

        return ResidualModule(
            self.channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.activation
        )
