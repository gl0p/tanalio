# base_model_node.py
import torch
import torch.nn as nn
from sockets import socketio  # for UI sync if needed

class BaseModelNode:
    def __init__(self):
        self.input_tensor_shape = None  # e.g. [batch, channels, H, W]
        self.in_features = None
        self.out_features = None
        self.activation = "none"  # default, can be 'relu', 'sigmoid', etc.
        self.lock_in_features = False  # prevents auto-override
        self.graph_node_id = None

    def set_active(self, is_active=True):
        if self.graph_node_id is not None:
            socketio.emit("node_active" if is_active else "node_inactive", {
                "node_id": self.graph_node_id
            })

    def set_input_shape(self, tensor):
        """
        Set input shape from an upstream tensor. This will calculate in_features.
        Supports both 4D images and 2D flat vectors.
        """
        print(f"input tensor is {tensor.shape}")
        if not isinstance(tensor, torch.Tensor):
            return
        shape = list(tensor.shape)
        if len(shape) >= 2:
            self.input_tensor_shape = shape[1:]  # remove batch dim
            self.in_features = int(torch.prod(torch.tensor(self.input_tensor_shape)))
            print(f"🧠 {self.__class__.__name__} input_shape = {self.input_tensor_shape}, in_features = {self.in_features}")

    def on_input_connected(self, input_index, source_node):
        """
        Called by executor when this node is connected to a previous node.
        """
        if not self.lock_in_features and hasattr(source_node, "out_features"):
            self.in_features = source_node.out_features
            print(f"🔄 Auto-updated in_features = {self.in_features} from {source_node.__class__.__name__}")

            # 🔁 UI sync (optional)
            if hasattr(self, "graph_node_id"):
                socketio.emit("property_update", {
                    "node_id": self.graph_node_id,
                    "property": "in_features",
                    "value": self.in_features
                })

    def get_activation(self):
        """
        Maps string → activation layer.
        """
        act = self.activation.lower()
        if act == "relu":
            return nn.ReLU()
        elif act == "sigmoid":
            return nn.Sigmoid()
        elif act == "tanh":
            return nn.Tanh()
        elif act == "softmax":
            return nn.Softmax(dim=1)
        else:
            return nn.Identity()

    def get_layer(self):
        """
        This must be implemented by child classes to return nn.Module.
        """
        raise NotImplementedError("Subclasses must implement get_layer()")

    def emit_update(self, prop, value):
        if self.graph_node_id is not None:
            socketio.emit("property_update", {
                "node_id": self.graph_node_id,
                "property": prop,
                "value": value
            })

    def compute_conv2d_output_shape(self, H, W, kernel_size, stride, padding):
        H_out = (H + 2 * padding - kernel_size) // stride + 1
        W_out = (W + 2 * padding - kernel_size) // stride + 1
        return H_out, W_out


def mark_active(fn):
    def wrapper(self, *args, **kwargs):
        self.set_active(True)
        try:
            return fn(self, *args, **kwargs)
        finally:
            self.set_active(False)
    return wrapper
