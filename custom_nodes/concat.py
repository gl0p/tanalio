import torch
from base_model_node import BaseModelNode, mark_active
from node_registry import register_node
from sockets import socketio

@register_node(name="Concat", category="Operations")
class ConcatNode(BaseModelNode):
    inputs = [
        {"name": "input_0", "type": "tensor"},
        {"name": "input_1", "type": "tensor"}
    ]
    outputs = [
        {"name": "tensor", "type": "tensor"}
    ]
    widgets = [
        {"type": "number", "name": "dim", "value": 1}
    ]
    size = [210, 70]

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def on_input_connected(self, input_index, source_node):
        # Skip calculating in_features — shape is determined dynamically
        super().on_input_connected(input_index, source_node)

    @mark_active
    def get_layer(self):
        def concat_fn(*tensors):
            try:
                base_shape = list(tensors[0].shape)
                for i, t in enumerate(tensors[1:], start=1):
                    if len(t.shape) != len(base_shape):
                        socketio.emit("node_error", {"node_id": self.graph_node_id})
                        socketio.emit("toast", {
                            "message": f"⚠️ Concat shape mismatch: expected rank {len(base_shape)}, got {len(t.shape)} at input {i}"
                        })
                        raise ValueError("Tensor rank mismatch.")

                    for dim_idx in range(len(base_shape)):
                        if dim_idx == self.dim:
                            continue
                        if t.shape[dim_idx] != base_shape[dim_idx]:
                            socketio.emit("node_error", {"node_id": self.graph_node_id})
                            socketio.emit("toast", {
                                "message": f"⚠️ Shape mismatch at dim {dim_idx}: expected {base_shape[dim_idx]}, got {t.shape[dim_idx]}"
                            })
                            raise ValueError("Tensor shape mismatch for concat.")

                return torch.cat(tensors, dim=self.dim)
            except Exception as e:
                print(f"❌ Concat failed: {e}")
                raise

        return concat_fn
