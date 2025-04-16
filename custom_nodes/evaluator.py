import torch
import torch.nn as nn
from sockets import socketio
from node_registry import register_node  # if using dynamic node loading

@register_node(name="Test Evaluator", category="Evaluators", tags={"run_after_train": True})
class TestEvaluatorNode:
    inputs = [
        {"name": "model", "type": "model_out"},
        {"name": "test", "type": "test"}
    ]

    outputs = []

    widgets = []

    size = [220, 100]

    def __init__(self):
        self.model = None
        self.test = None
        self.task_type = "classification"
        self.graph_node_id = None
        self.properties = {}

    def build(self):
        print("🚀 TestEvaluatorNode build() triggered")
        socketio.emit("node_active", {"node_id": self.graph_node_id})
        socketio.emit("node_clear_error", {"node_id": self.graph_node_id})

        if self.model is None or self.test is None:
            socketio.emit("node_error", {"node_id": self.graph_node_id})
            socketio.emit("toast", {"message": "❌ Missing model or test data"})
            return {"test_loss": None, "test_accuracy": None}

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        is_classification = self.task_type == "classification"
        loss_fn = nn.CrossEntropyLoss() if is_classification else nn.MSELoss()

        with torch.no_grad():
            for i, batch in enumerate(self.test):
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, y = batch
                elif isinstance(batch, dict):
                    keys = list(batch.keys())
                    x_key = next((k for k in keys if "in" in k.lower() or "x" in k.lower()), keys[0])
                    y_key = next((k for k in keys if "label" in k.lower() or "y" in k.lower()), keys[1] if len(keys) > 1 else keys[0])
                    x, y = batch[x_key], batch[y_key]
                elif isinstance(batch, torch.Tensor):
                    x = batch
                    y = None
                else:
                    socketio.emit("toast", {
                        "message": f"❌ Unknown test batch format at index {i}"
                    })
                    print(f"❌ Unhandled batch: {batch}")
                    raise ValueError("Unknown batch format")

                outputs = self.model(x)

                if y is not None:
                    if y.dim() == 3 and y.size(2) == 1:
                        y = y.squeeze(2)

                    loss = loss_fn(outputs, y)
                    total_loss += loss.item()

                    if is_classification:
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == y).sum().item()
                        total += y.size(0)
                else:
                    total_loss += 0.0

        avg_loss = total_loss / len(self.test)
        acc = (correct / total) * 100 if is_classification and total > 0 else None

        # 🧠 Store props for UI
        self.properties = {
            "test_loss": str(round(avg_loss, 4)),
            "test_accuracy": f"{acc:.2f}%" if acc is not None else "N/A"
        }

        # 🔔 Notify UI
        msg = f"✅ Test Loss: {avg_loss:.4f}"
        if acc is not None:
            msg += f" | Accuracy: {acc:.2f}%"

        if acc is not None:
            print(f"Evaluation results: Loss = {avg_loss} Accuracy = {float(acc)}")
        else:
            print(f"Evaluation results: Loss = {avg_loss} (no accuracy for regression)")

        socketio.emit("toast", {"message": msg})
        socketio.emit("node_inactive", {"node_id": self.graph_node_id})
        socketio.emit("test_accuracy_result", {
            "node_id": self.graph_node_id,
            "accuracy": float(acc) if acc is not None else None,
            "loss": float(avg_loss)
        })

        return {
            "test_loss": avg_loss,
            "test_accuracy": acc
        }
