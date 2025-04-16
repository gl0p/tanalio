import torch.nn as nn
from base_preprocess_node import BasePreprocessNode
from node_registry import register_node
from sockets import socketio

@register_node(name="AutoPermute", category="Preprocessing", tags={"run_early": True})
class AutoPermute(BasePreprocessNode):
    inputs = [
        {"name": "in", "type": "tensor"}
    ]
    outputs = [
        {"name": "out", "type": "tensor"}
    ]
    widgets = [
        {
            "type": "text",
            "name": "dims_str",
            "value": "0, 3, 1, 2",
            "options": {"placeholder": "e.g. 0, 3, 1, 2"}
        }
    ]
    size = [210, 70]

    def __init__(self, dims_str="0, 3, 1, 2"):
        super().__init__()
        self.dims_str = dims_str
        self.dims = self.parse_dims(dims_str)

    def parse_dims(self, s):
        try:
            parsed = tuple(int(x.strip()) for x in s.split(","))
            return parsed
        except Exception as e:
            socketio.emit("toast", {
                "message": f"⚠️ Invalid permute dims: {s}. Falling back to identity."
            })
            return tuple(range(4))  # fallback

    def set_input_shape(self, tensor):
        # 👇 Let base class do its thing first (sets input_shape, out_features, etc.)
        super().set_input_shape(tensor)

        # 👀 Auto-suggest dim order based on rank
        shape = self.input_tensor_shape
        if shape is None:
            return

        rank = len(shape)
        default_perm = tuple(range(rank))  # identity by default

        # 🎯 Auto suggest NHWC → NCHW if it looks like image shape
        if rank == 4 and shape[1] > shape[2] and shape[1] > shape[3]:
            default_perm = (0, 2, 3, 1)  # NHWC to NCHW

        self.dims = default_perm
        self.dims_str = ", ".join(str(i) for i in default_perm)

        # 🧠 Tell the frontend
        socketio.emit("property_update", {
            "node_id": self.graph_node_id,
            "property": "dims_str",
            "value": self.dims_str
        })

        print(f"🧠 AutoPermute suggested dims {self.dims_str} for shape {shape}")

    def get_layer(self):
        class PermuteLayer(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.dims = dims

            def forward(self, x):
                return x.permute(*self.dims)

        return PermuteLayer(self.dims)
