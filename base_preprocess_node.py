# base_preprocess_node.py

from sockets import socketio

class BasePreprocessNode:
    def __init__(self):
        self.input_shape = None
        self.out_features = None
        self.graph_node_id = None  # must be set externally
        self.input_tensor_shape = None
        self.graph_node_id = None

    def set_active(self, is_active=True):
        if self.graph_node_id is not None:
            socketio.emit("node_active" if is_active else "node_inactive", {
                "node_id": self.graph_node_id
            })

    def set_input_shape(self, tensor):
        self.input_tensor_shape = tensor.shape if hasattr(tensor, "shape") else tensor

        if self.input_tensor_shape is None:
            socketio.emit("toast", {
                "message": f"⚠️ Sample_tensor is None. Make sure you are loading the folder that contains train, test and val."
            })

        shape = list(tensor.shape)
        if len(shape) >= 2:
            self.input_shape = shape[1:]  # drop batch dim
        else:
            self.input_shape = shape

        self.out_features = 1
        for dim in self.input_shape:
            self.out_features *= dim

        print(f"🧠 {self.__class__.__name__} detected input shape {self.input_shape}, out_features = {self.out_features}")

        socketio.emit("property_update", {
            "node_id": self.graph_node_id,
            "property": "out_features",
            "value": self.out_features
        })

def mark_active(fn):
    def wrapper(self, *args, **kwargs):
        self.set_active(True)
        try:
            return fn(self, *args, **kwargs)
        finally:
            self.set_active(False)
    return wrapper