import torch.nn as nn
from sockets import socketio
from node_registry import register_node

@register_node(name="Output Layer", category="Model")
class OutputLayer:
    inputs = [
        {"name": "final_layer", "type": "tensor"}
    ]
    outputs = [
        {"name": "model", "type": "model"}
    ]
    widgets = [
        {"type": "text", "name": "model_name", "value": "MyModel"},
        {"type": "combo", "name": "activation", "value": "softmax", "options": {"values": ["softmax", "sigmoid", "none"]}},
        {"type": "number", "name": "in_features", "value": 64, "options": {"min": 1, "max": 2048}},
        {"type": "number", "name": "out_features", "value": 10, "options": {"min": 1, "max": 2048}},
    ]
    size = [220, 120]

    def __init__(self, model_name="MyModel", activation="softmax", in_features=64, out_features=10, type="sequential", num_classes=None):
        self.model_name = model_name
        self.activation = activation
        self.in_features = in_features
        self.out_features = out_features
        self.type = type
        self.num_classes = num_classes

        self.graph_nodes = {}
        self.links = {}
        self.graph_node_id = None
        self.graph_nodes_data = {}
        self.final_node_id = None

    def build(self):
        print("🔧 Building Output Layer and preceding layers...")
        socketio.emit("node_active", {"node_id": self.graph_node_id})
        visited = set()
        layers = []

        task_type = getattr(self, "task_type", "classification")
        print(f"🧠 OutputLayer received task_type: {task_type}")

        if task_type == "regression":
            if self.activation != "none" or self.out_features != 1:
                print("🔄 Auto-switching OutputLayer to regression mode")
                self.out_features = 1
                self.activation = "none"
                socketio.emit("property_update", {"node_id": self.graph_node_id, "property": "out_features", "value": 1})
                socketio.emit("property_update", {"node_id": self.graph_node_id, "property": "activation", "value": "none"})

        elif task_type == "classification":
            if self.num_classes:
                self.out_features = self.num_classes
                socketio.emit("property_update", {"node_id": self.graph_node_id, "property": "out_features", "value": self.out_features})
                socketio.emit("toast", {"message": f"🎯 OutputLayer auto-set out_features = {self.out_features} from num_classes"})
                print(f"🎯 Auto-set out_features = {self.out_features}")

            if self.activation != "softmax":
                self.activation = "softmax"
                socketio.emit("property_update", {"node_id": self.graph_node_id, "property": "activation", "value": "softmax"})

        def walk_back(node_id):
            if node_id in visited:
                return
            visited.add(node_id)

            node_dict = self.graph_nodes_data.get(node_id)
            for i, input in enumerate(node_dict.get("inputs", [])):
                link_id = input.get("link")
                if link_id is not None:
                    prev_id = self.links[link_id][0]
                    walk_back(prev_id)

                    target_node = self.graph_nodes[node_id]
                    source_node = self.graph_nodes[prev_id]
                    if hasattr(target_node, "on_input_connected"):
                        target_node.on_input_connected(i, source_node)
                        print(f"🔌 Triggered on_input_connected on {target_node.__class__.__name__}")

            node = self.graph_nodes.get(node_id)
            if isinstance(node, nn.Module):
                layers.append(node)
            elif hasattr(node, "get_layer"):
                layer = node.get_layer()
                if isinstance(layer, nn.Module):
                    layers.append(layer)

        walk_back(self.final_node_id)

        # Add final output projection layer
        layers.append(nn.Linear(self.in_features, self.out_features))

        if self.activation == "softmax":
            layers.append(nn.Softmax(dim=1))
        elif self.activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif self.activation == "relu":
            layers.append(nn.ReLU())

        model = nn.Sequential(*layers)
        print("📦 Final model structure:")
        print(model)
        socketio.emit("node_inactive", {"node_id": self.graph_node_id})
        return model
