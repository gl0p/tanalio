import torch
import torch.nn as nn
from base_model_node import BaseModelNode
from node_registry import register_node

@register_node(name="LSTM", category="Model")
class LSTM(BaseModelNode):
    inputs = [
        {"name": "in", "type": "tensor"}
    ]
    outputs = [
        {"name": "out", "type": "tensor"}
    ]
    widgets = [
        {"type": "number", "name": "input_size", "value": 128, "options": {"min": 1, "max": 1024}},
        {"type": "number", "name": "hidden_size", "value": 64, "options": {"min": 1, "max": 1024}},
        {"type": "number", "name": "num_layers", "value": 1, "options": {"min": 1, "max": 10}},
        {"type": "toggle", "name": "bidirectional", "value": False}
    ]
    size = [200, 120]

    def __init__(self, input_size=128, hidden_size=64, num_layers=1, bidirectional=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

    def get_layer(self):
        actual_input_size = self.in_features or self.input_size
        self.out_features = self.hidden_size
        self.emit_update("out_features", self.out_features)
        self.emit_update("input_size", actual_input_size)

        class LSTMWrapper(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    bidirectional=bidirectional
                )

            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # [B, 1, F]
                out, _ = self.lstm(x)
                return out[:, -1, :]  # use last timestep only

        return LSTMWrapper(
            actual_input_size,
            self.hidden_size,
            self.num_layers,
            self.bidirectional
        )
