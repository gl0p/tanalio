import torch
import torch.nn as nn
from base_model_node import BaseModelNode, mark_active
from node_registry import register_node
from sockets import socketio

@register_node(name="Conv2D", category="Model")
class Conv2DLayer(BaseModelNode):
    inputs = [{"name": "in", "type": "tensor"}]
    outputs = [{"name": "out", "type": "tensor"}]
    widgets = [
        {"type": "number", "name": "in_channels", "value": 3, "options": {"min": 1, "max": 512}},
        {"type": "number", "name": "out_channels", "value": 16, "options": {"min": 1, "max": 512}},
        {"type": "number", "name": "kernel_size", "value": 3, "options": {"min": 1, "max": 11}},
        {"type": "number", "name": "stride", "value": 1, "options": {"min": 1, "max": 5}},
        {"type": "number", "name": "padding", "value": 1, "options": {"min": 0, "max": 5}},
        {"type": "combo", "name": "activation", "value": "relu", "options": {"values": ["relu", "sigmoid", "none"]}},
    ]
    size = [200, 160]

    def __init__(self, in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

    def set_input_shape(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            return

        shape = list(tensor.shape)
        if len(shape) == 4:  # Expecting [B, C, H, W]
            self.input_tensor_shape = shape[1:]
            self.in_channels = shape[1]
            self.emit_update("in_channels", self.in_channels)

            H, W = shape[2], shape[3]
            H_out, W_out = self.compute_conv2d_output_shape(H, W, self.kernel_size, self.stride, self.padding)
            self.out_features = self.out_channels * H_out * W_out
            self.output_tensor_shape = [self.out_channels, H_out, W_out]

            self.emit_update("out_features", self.out_features)

            print(f"🧠 Conv2D received input: {shape} → out_shape = {self.output_tensor_shape}, out_features = {self.out_features}")
        else:
            print(f"⚠️ Conv2D expected 4D input but got shape: {shape}")

    @mark_active
    def get_layer(self):
        layers = [
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding
            ),
            self.get_activation()
        ]
        return nn.Sequential(*layers)
