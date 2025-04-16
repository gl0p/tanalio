# batch_norm_1d.py
import torch.nn as nn
from base_model_node import BaseModelNode
from node_registry import register_node

@register_node(name="BatchNorm1d", category="Model")
class BatchNorm1dNode(BaseModelNode):
    inputs = [
        {"name": "in", "type": "tensor"}
    ]
    outputs = [
        {"name": "out", "type": "tensor"}
    ]
    widgets = [
        {
            "type": "number",
            "name": "num_features",
            "value": 64,
            "options": {"min": 1, "max": 4096, "step": 1}
        },
        {
            "type": "combo",
            "name": "affine",
            "value": "on",
            "options": {"values": ["on", "off"]}
        },
        {
            "type": "combo",
            "name": "track_running_stats",
            "value": "on",
            "options": {"values": ["on", "off"]}
        }
    ]
    size = [200, 100]

    def __init__(self,
                 num_features=64,
                 affine="on",
                 track_running_stats="on"):
        super().__init__()
        self.num_features = num_features
        self.affine = affine == "on"
        self.track_running_stats = track_running_stats == "on"
        self.out_features = num_features  # same as in_features

    def get_layer(self):
        self.out_features = self.num_features  # propagate shape
        self.emit_update("out_features", self.out_features)
        return nn.BatchNorm1d(
            num_features=self.num_features,
            affine=self.affine,
            track_running_stats=self.track_running_stats
        )
