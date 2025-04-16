import torch
import torch.nn as nn
from base_preprocess_node import BasePreprocessNode
from node_registry import register_node

@register_node(name="AutoCast", category="Preprocessing", tags={"run_early": True})
class AutoCast(BasePreprocessNode):
    inputs = [
        { "name": "in", "type": "tensor" }
    ]
    outputs = [
        { "name": "out", "type": "tensor" }
    ]
    widgets = [
        {
            "type": "combo",
            "name": "dtype",
            "value": "float32",
            "options": { "values": ["float16", "float32", "float64"] }
        }
    ]
    size = [210, 70]

    def __init__(self, dtype="float32"):
        super().__init__()
        self.dtype = dtype

    def get_layer(self):
        class CastLayer(nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                torch_dtype = getattr(torch, self.dtype, None)
                if torch_dtype is None:
                    raise ValueError(f"Invalid dtype: {self.dtype}")
                return x.to(dtype=torch_dtype)

        return CastLayer(self.dtype)
