import torch
import torch.nn as nn
from base_model_node import BaseModelNode, mark_active
from sockets import socketio
from node_registry import register_node

@register_node(name="Add", category="Operations")
class AddNode(BaseModelNode):
    inputs = [
        {"name": "input_0", "type": "tensor"},
        {"name": "input_1", "type": "tensor"},
    ]

    outputs = [
        {"name": "tensor", "type": "tensor"}
    ]

    widgets = []  # No widgets needed

    size = [180, 80]

    def __init__(self):
        super().__init__()
        self.output_shape = None

    @mark_active
    def get_layer(self):
        def add_fn(*inputs):
            if len(inputs) < 2:
                raise ValueError("AddNode requires at least 2 inputs.")

            base_shape = inputs[0].shape
            for idx, tensor in enumerate(inputs[1:], start=1):
                if tensor.shape != base_shape:
                    socketio.emit("node_error", {"node_id": self.graph_node_id})
                    socketio.emit("toast", {
                        "message": f"⚠️ Shape mismatch at input {idx}: expected {base_shape}, got {tensor.shape}"
                    })
                    raise ValueError(f"Shape mismatch at input {idx}: expected {base_shape}, got {tensor.shape}")

            return torch.stack(inputs, dim=0).sum(dim=0)

        return add_fn
