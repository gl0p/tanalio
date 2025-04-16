import torch.nn as nn
from base_model_node import BaseModelNode
from node_registry import register_node

@register_node(name="Dropout", category="Model")
class DropoutLayer(BaseModelNode):
    inputs = [
        {"name": "in", "type": "tensor"}
    ]
    outputs = [
        {"name": "out", "type": "tensor"}
    ]
    widgets = [
        {
            "type": "number",
            "name": "dropout_rate",
            "value": 0.5,
            "options": {"min": 0.0, "max": 1.0, "step": 0.05}
        }
    ]
    size = [180, 80]

    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate

    def get_layer(self):
        # Passthrough features
        self.out_features = self.in_features
        self.emit_update("out_features", self.out_features)
        return nn.Dropout(p=self.dropout_rate)

    def build(self):
        # Optional if this node is ever used in config-only builds
        return {
            "out_features": self.out_features
        }
