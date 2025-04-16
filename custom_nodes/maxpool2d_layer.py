import torch.nn as nn
from base_model_node import BaseModelNode
from node_registry import register_node
from sockets import socketio
import torch

@register_node(name="MaxPool2D", category="Model")
class MaxPool2DLayer(BaseModelNode):
    inputs = [{"name": "in", "type": "tensor"}]
    outputs = [{"name": "out", "type": "tensor"}]
    widgets = [
        {"type": "number", "name": "kernel_size", "value": 2, "options": {"min": 1, "max": 4}},
        {"type": "number", "name": "stride", "value": 2, "options": {"min": 1, "max": 4}}
    ]
    size = [180, 80]

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def set_input_shape(self, tensor):
        super().set_input_shape(tensor)

        if self.input_tensor_shape and len(self.input_tensor_shape) >= 3:
            C, H, W = self.input_tensor_shape[-3:]
            H_out, W_out = self.compute_conv2d_output_shape(H, W, self.kernel_size, self.stride, padding=0)
            new_shape = [C, H_out, W_out]
            self.input_tensor_shape = new_shape
            self.out_features = int(torch.prod(torch.tensor(new_shape)))

            socketio.emit("property_update", {
                "node_id": self.graph_node_id,
                "property": "out_features",
                "value": self.out_features
            })

            print(f"📐 MaxPool2D updated shape → {new_shape} → out_features = {self.out_features}")

    def get_layer(self):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size=self.kernel_size, stride=self.stride)
        )
