import torch
import numpy as np
import os
from sockets import socketio
from node_registry import register_node

@register_node(name="Single Predict", category="Predictors", tags={"run_after_train": True})
class SinglePredictNode:
    inputs = [
        {"name": "model", "type": "model_out"}
    ]
    outputs = [
        {"name": "prediction", "type": "number"}
    ]

    widgets = [
        # 🧠 These will get dynamically injected based on input features
        {"type": "button", "name": "Run Prediction", "value": "", "callback": "run_prediction"},
        {"type": "text", "name": "prediction", "value": "n/a"}
    ]

    size = [260, 220]

    def __init__(self):
        self.graph_node_id = None
        self.input_features = []
        self.feature_names = []
        self.normalize_input = True
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        self.input_dim = None
        self.task_type = "regression"
        self.model = None
        self.title = "Single Predict"
        # self.input_features = ["20190509",	"59.06",	"59.11",	"55.94",	"56.74",	"124125121"]
        self.input_features = []

    def set_feature_info(self, feature_names, input_mean=None, input_std=None, output_mean=None, output_std=None):
        self.feature_names = feature_names
        self.input_mean = np.array(input_mean) if input_mean is not None else None
        self.input_std = np.array(input_std) if input_std is not None else None
        self.output_mean = output_mean
        self.output_std = output_std
        self.input_dim = len(feature_names)
        socketio.emit("dynamic_widget_update", {
            "node_id": self.graph_node_id,
            "feature_names": self.feature_names
        })

    def on_widget_event(self, event_type, payload):
        if event_type == "run_prediction":
            # Rebuild input_features from current widget values
            self.input_features = [payload.get(f"feature_{name}", 0) for name in self.feature_names]
            result = self.run_prediction(self.input_features)
            socketio.emit("single_predict_result", {
                "node_id": self.graph_node_id,
                "result": result
            })

    def run_prediction(self, input_features):
        socketio.emit("node_active", {"node_id": self.graph_node_id})
        try:
            model = self._resolve_model()

            # ✅ Use the passed-in input_features
            x = np.array(input_features, dtype=np.float32).reshape(1, -1)
            if self.normalize_input and self.input_mean is not None:
                x = (x - self.input_mean) / (self.input_std + 1e-8)
            x_tensor = torch.tensor(x, dtype=torch.float32)

            with torch.no_grad():
                prediction = model(x_tensor)

            if self.task_type == "regression" and self.output_mean is not None:
                prediction = prediction * self.output_std + self.output_mean

            prediction_value = prediction.squeeze().tolist()
            self._update_prediction(prediction_value)

            print(f"🧠 Prediction: {prediction_value}")
            socketio.emit("toast", {"message": f"🧠 Prediction: {prediction_value}"})
            return prediction_value

        except Exception as e:
            print("❌ Prediction error:", e)
            socketio.emit("toast", {"message": f"❌ Prediction failed: {str(e)}"})
            return None

        finally:
            socketio.emit("node_inactive", {"node_id": self.graph_node_id})

    def _update_prediction(self, value):
        socketio.emit("property_update", {
            "node_id": self.graph_node_id,
            "property": "prediction",
            "value": str(round(float(value), 4)) if isinstance(value, (int, float)) else str(value)
        })

    def _resolve_model(self):
        model_out = self.model  # populated via input binding
        if isinstance(model_out, torch.nn.Module):
            return model_out
        elif isinstance(model_out, dict) and isinstance(model_out.get("model"), torch.nn.Module):
            return model_out["model"]
        elif isinstance(model_out, str) and model_out.endswith(".pt") and os.path.exists(model_out):
            if self.input_dim is None:
                raise Exception("input_dim required to load model from file.")
            return self._load_model_from_file(model_out)
        raise Exception("Unsupported model format")

    def _load_model_from_file(self, path):
        # Example fallback, replace with user-defined model loading logic
        from models import YourModelClass
        model = YourModelClass(input_dim=self.input_dim)
        model.load_state_dict(torch.load(path))
        return model

    def build(self, model_out=None, task_type="regression"):
        print(f"🔧 SinglePredictNode.build called")
        self.model = model_out
        self.task_type = task_type
        socketio.emit("node_active", {"node_id": self.graph_node_id})
        socketio.emit("node_inactive", {"node_id": self.graph_node_id})
        return None
