import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sockets import socketio
from utils.loss_registry import LOSS_FUNCTIONS
from node_registry import register_node

@register_node(name="Trainer", category="Training", tags={"register_runtime": True})
class TrainerNode:
    inputs = [
        {"name": "model", "type": "model"},
        {"name": "train", "type": "train"},
        {"name": "val", "type": "val"},
        {"name": "hyperparams", "type": "dict"}
    ]
    outputs = [
        {"name": "trained_model", "type": "model_out"}
    ]
    widgets = []
    size = [220, 120]

    def __init__(self):
        self.model = None
        self.train = None
        self.val = None
        self.hyperparams = {}
        self.graph_nodes = {}
        self.sample_tensor = []
        self.losses = []
        self.epochs = []
        self.graph_node_id = None
        self._pause = False
        self._stop = False
        self.task_type = None

    def pause(self):
        self._pause = True
        print("⏸️ Training paused")

    def resume(self):
        self._pause = False
        print("▶️ Training resumed")

    def stop(self):
        self._stop = True
        print("🛑 Training stopped")

    def _handle_pause(self):
        while self._pause:
            socketio.sleep(0.1)

    def build(self):
        print(f"👀 TrainerNode.build() → self.train: {self.train}, type: {type(self.train)}")
        print(f"🎯 TrainerNode ID: {self.graph_node_id}")

        socketio.emit("node_clear_error", {"node_id": self.graph_node_id})
        socketio.emit("node_active", {"node_id": self.graph_node_id})

        if isinstance(self.train, dict):
            self.sample_tensor = self.train.get("sample_tensor")
            self.train = self.train.get("train")
            self.val = self.val.get("val") if isinstance(self.val, dict) else self.val
            self.task_type = self.train.get("task_type", "classification")

        for inputs, labels in self.train:
            print("🧪 Input shape:", inputs.shape)
            if hasattr(self.model, "sample_tensor"):
                self.model.sample_tensor = inputs
            break

        if not isinstance(self.train, DataLoader) or not isinstance(self.val, DataLoader):
            raise Exception("⚠️ Trainer expected train/val to be DataLoaders!")

        if not isinstance(self.model, nn.Module):
            raise Exception("⚠️ No valid model provided to Trainer")

        print("✅ Model and DataLoaders received. Training begins...")
        print(f"🔍 Received hyperparams: {self.hyperparams}")

        # 🔧 Extract hyperparameters
        epochs = self.hyperparams.get("epochs", 1)
        lr = self.hyperparams.get("learning_rate", 0.001)
        loss_name = self.hyperparams.get("loss", "auto")
        optimizer_name = self.hyperparams.get("optimizer", "adam")
        use_early_stopping = self.hyperparams.get("use_early_stopping", "off") == "on"
        patience = self.hyperparams.get("early_stopping_patience", 5)

        # ⚠️ Incompatible loss check
        if loss_name == "mse" and getattr(self.model, "activation", "") == "softmax":
            socketio.emit("node_inactive", {"node_id": self.graph_node_id})
            socketio.emit("node_error", {"node_id": self.graph_node_id})
            socketio.emit("toast", {
                "message": "❌ MSE loss is not compatible with softmax activation. Use 'none' or switch to cross_entropy."
            })
            raise ValueError("Incompatible loss/activation: MSE with softmax")

        # 🎯 Auto or registered loss
        if loss_name == "auto":
            task_type = getattr(self.train.dataset, "task_type", "classification")
            loss_fn = nn.MSELoss() if task_type == "regression" else nn.CrossEntropyLoss()
        else:
            loss_class = LOSS_FUNCTIONS.get(loss_name)
            if not loss_class:
                raise ValueError(f"❌ Unknown loss: {loss_name}")
            loss_fn = loss_class()

        optimizer_cls = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD,
            "rmsprop": torch.optim.RMSprop
        }.get(optimizer_name, torch.optim.Adam)

        optimizer = optimizer_cls(self.model.parameters(), lr=lr)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            if self._stop:
                print("🛑 Training manually stopped.")
                break
            while self._pause:
                self._handle_pause()

            self.model.train()
            running_loss = 0.0

            for inputs, labels in self.train:
                if self._stop:
                    break
                while self._pause:
                    self._handle_pause()

                optimizer.zero_grad()

                if isinstance(labels, torch.Tensor) and labels.dim() == 3 and labels.size(2) == 1:
                    labels = labels.squeeze(2)

                outputs = self.model(inputs)
                try:
                    if isinstance(labels, list) and isinstance(labels[0], dict):
                        loss = loss_fn(outputs, labels)
                    else:
                        loss = loss_fn(outputs, labels)
                except Exception as e:
                    socketio.emit("toast", {
                        "message": f"❌ Loss function mismatch. Change loss in Hyperparameter node."
                    })
                    print(f"❌ Loss error → outputs: {outputs.shape}, labels: {getattr(labels, 'shape', type(labels))}")
                      # optionally raise again to crash or remove this to silently skip

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            if self._stop:
                print("🛑 Training manually stopped.")
                break
            while self._pause:
                self._handle_pause()

            avg_loss = running_loss / len(self.train)
            self.epochs.append(epoch + 1)
            self.losses.append(avg_loss)
            print(f"📉 Epoch [{epoch + 1}] Train Loss: {avg_loss:.4f}")

            # 🔍 Validation
            val_loss = 0.0
            correct, total = 0, 0
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.val:
                    if self._stop:
                        print("🛑 Training manually stopped.")
                        break
                    while self._pause:
                        self._handle_pause()

                    if isinstance(labels, torch.Tensor) and labels.dim() == 3 and labels.size(2) == 1:
                        labels = labels.squeeze(2)

                    outputs = self.model(inputs)
                    if isinstance(labels, list) and isinstance(labels[0], dict):
                        loss = loss_fn(outputs, labels)
                    else:
                        loss = loss_fn(outputs, labels)

                    val_loss += loss.item()
                    if isinstance(loss_fn, nn.CrossEntropyLoss):
                        preds = torch.argmax(outputs, dim=1)
                        correct += (preds == labels).sum().item()
                        total += labels.size(0)

            avg_val_loss = val_loss / len(self.val)
            acc = (correct / total) * 100 if total > 0 else None
            print(f"🧪 Val Loss: {avg_val_loss:.4f}", end="")
            if acc is not None:
                print(f" | Accuracy: {acc:.2f}%")
            else:
                print()

            socketio.emit("loss_update", {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "val_loss": avg_val_loss,
                "accuracy": acc
            })

            if use_early_stopping:
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f"⏳ EarlyStopping counter: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print("🛑 Early stopping triggered!")
                        break

            socketio.sleep(0)

        socketio.emit("node_inactive", {"node_id": self.graph_node_id})
        print("✅ Training complete!")
        return self.model
