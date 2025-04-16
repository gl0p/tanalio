import torch
from base_model_node import BaseModelNode
from node_registry import register_node
from sockets import socketio

@register_node(name="Residual Merge", category="Operations")
class ResidualMergeNode(BaseModelNode):
    inputs = [
        {"name": "main", "type": "tensor"},
        {"name": "skip", "type": "tensor"}
    ]
    outputs = [
        {"name": "tensor", "type": "tensor"}
    ]
    widgets = [
        {"type": "toggle", "name": "use_skip_connection", "value": True}
    ]
    size = [200, 80]

    def __init__(self, use_skip_connection=True):
        super().__init__()
        self.use_skip_connection = use_skip_connection

    def get_layer(self):
        def merge_fn(main, skip):
            if not self.use_skip_connection:
                return main

            if main.shape != skip.shape:
                socketio.emit("node_error", {"node_id": self.graph_node_id})
                socketio.emit("toast", {
                    "message": f"⚠️ Residual shape mismatch: main={main.shape}, skip={skip.shape}"
                })
                raise ValueError(f"Residual shape mismatch: main={main.shape}, skip={skip.shape}")

            return main + skip

        return merge_fn
