from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os
from utils.dataset_utils import validate_or_split_csv_dataset
from sockets import socketio
from node_registry import register_node
from graph_executor import find_downstream_predictors

@register_node(name="Load CSV", category="Loaders", tags={"run_early": True})
class LoadCSV:
    widgets = [
        {"type": "folder_picker", "name": "folder_path", "value": ""},
        {"type": "number", "name": "batch_size", "value": 32},
        {"type": "combo", "name": "normalize_input", "value": "on", "options": {"values": ["on", "off"]}},
        {"type": "combo", "name": "normalize_output", "value": "on", "options": {"values": ["on", "off"]}},
        {"type": "combo", "name": "target_column", "value": "", "options": {"values": []}},  # Dynamic options later
        {"type": "label", "name": "dataset_status", "value": "Not validated"},
    ]
    inputs = []
    outputs = [
        {"name": "train", "type": "train"},
        {"name": "val", "type": "val"},
        {"name": "test", "type": "test"},
        {"name": "sample_tensor", "type": "tensor"},
    ]
    def __init__(self, folder_path="", batch_size=32, file_list=None,
                 normalize_input="on", normalize_output="on", target_column=None):
        self.folder_path = folder_path
        self.batch_size = batch_size
        self.file_list = file_list or []
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.target_column = target_column
        self.columns = []
        self.progress = 0
        self.validation_status = "Pending"
        self.graph_node_id = None
        self.feature_mean = None
        self.feature_std = None
        self.output_mean = None
        self.output_std = None
        self.title = "Load CSV"

    def on_widget_event(self, event_type, payload):
        if event_type != "widget_event":
            return

        updated = False
        # ✅ Update folder if provided (usually when user picks folder)
        if "folder_path" in payload:
            self.folder_path = payload["folder_path"]
            self.file_list = payload.get("file_list", [])
            print(f"📂 Folder picked: {self.folder_path} with {len(self.file_list)} files")

            # Try to auto-validate (optional fallback)
            try:
                result = validate_or_split_csv_dataset(self.folder_path, self.file_list[0])
                self.columns = result.get("columns", [])
                self.target_column = self.columns[-1] if self.columns else None
                updated = True
            except Exception as e:
                print("❌ Folder validation failed:", e)

        # ✅ Update target column if user changes it
        if "target_column" in payload:
            self.target_column = payload["target_column"]
            updated = True

        if not updated:
            print("⚠️ No relevant widget update keys found in payload.")
            return

        # 🔄 Re-inject input features
        if not self.columns or not self.target_column:
            print("⚠️ Skipping predictor update: missing columns or target_column.")
            return

        input_features = [c for c in self.columns if c != self.target_column]

        predictors = find_downstream_predictors(self.graph_node_id)
        print("👀 Found downstream predictors:", [p.graph_node_id for p in predictors])
        for node in predictors:
            if hasattr(node, "set_feature_info"):
                node.set_feature_info(
                    feature_names=input_features,
                    input_mean=self.feature_mean,
                    input_std=self.feature_std,
                    output_mean=self.output_mean,
                    output_std=self.output_std
                )
                print(f"🔁 Live-injected input features to predictor {node.graph_node_id}")

    def build(self, task_type_override=None):
        socketio.emit("node_active", {"node_id": self.graph_node_id})
        socketio.emit("node_clear_error", {"node_id": self.graph_node_id})

        if not self.file_list:
            socketio.emit("toast", {"message": "⚠️ No CSV file selected"})
            raise ValueError("No CSV file selected")

        base_data_dir = os.path.abspath("datasets/csv")
        relative_path = os.path.join(base_data_dir, self.folder_path)
        print(f"📂 Validating CSV: {relative_path}")

        train_file = os.path.join(relative_path, "train.csv")
        val_file = os.path.join(relative_path, "val.csv")
        test_file = os.path.join(relative_path, "test.csv")

        if all(map(os.path.exists, [train_file, val_file, test_file])):
            socketio.emit("toast", {"message": "✅ CSV already split."})
            try:
                df = pd.read_csv(train_file, nrows=1)
                columns = list(df.columns)
            except Exception as e:
                print("⚠️ Could not read header:", e)
                columns = []
            result = {"status": "validated", "progress": 100, "columns": columns}
        else:
            result = validate_or_split_csv_dataset(relative_path, self.file_list[0])
            print("🔎 CSV dataset result:", result)

        self.columns = result.get("columns", [])
        self.progress = result["progress"]
        self.validation_status = result["status"]

        if result["status"] == "split":
            socketio.emit("toast", {"message": "✅ CSV cleaned and split."})
        elif result["status"] == "error":
            raise ValueError(result.get("message", "Unknown CSV error"))

        dataloaders = {}
        sample_tensor = None

        for split in ["train", "val", "test"]:
            path = os.path.join("datasets/csv", self.folder_path, f"{split}.csv")
            if os.path.exists(path):
                dataset = CSVDataset(
                    path,
                    task_type_override=task_type_override,
                    normalize_input=self.normalize_input == "on",
                    normalize_output=self.normalize_output == "on",
                    target_column=self.target_column
                )
                dataloaders[split] = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Sample tensor for shape propagation
        if "train" in dataloaders:
            try:
                batch = next(iter(dataloaders["train"]))
                x = batch[0] if isinstance(batch, tuple) else batch
                sample_tensor = x[0]
                print(f"🧪 Sample input tensor shape: {sample_tensor.shape}")
            except Exception as e:
                print("❌ Sample tensor error:", str(e))
                socketio.emit("node_error", {"node_id": self.graph_node_id})
                socketio.emit("toast", {"message": f"❌ Failed to extract sample tensor: {e}"})

        socketio.emit("node_inactive", {"node_id": self.graph_node_id})

        # Determine task type and number of classes
        dataset = dataloaders.get("train").dataset if "train" in dataloaders else None
        task_type = task_type_override or (
            "classification" if dataset and getattr(dataset, "is_classification", False) else "regression"
        )
        num_classes = getattr(dataset, "num_classes", None)

        # Save normalization stats
        if dataset:
            self.feature_mean = getattr(dataset, "feature_mean", None)
            self.feature_std = getattr(dataset, "feature_std", None)
            self.output_mean = getattr(dataset, "labels_mean", None)
            self.output_std = getattr(dataset, "labels_std", None)
        print(f"📦 LoadCSV.build() → returning:")
        print(f"   train: {type(train_file)} | val: {type(val_file)} | sample_tensor: {type(sample_tensor)}")

        return {
            "train": dataloaders.get("train"),
            "val": dataloaders.get("val"),
            "test": dataloaders.get("test"),
            "sample_tensor": sample_tensor,
            "task_type": task_type,
            "num_classes": num_classes,
            "columns": self.columns,
            "input_mean": self.feature_mean,
            "input_std": self.feature_std,
            "output_mean": self.output_mean,
            "output_std": self.output_std,
        }


class CSVDataset(Dataset):
    def __init__(self, path, task_type_override=None, normalize_input=False, normalize_output=False, target_column=None):
        self.data = pd.read_csv(path)
        self.target_column = target_column or self.data.columns[-1]
        self.task_type = "regression"
        self.is_classification = False
        self.num_classes = None

        try:
            label_vals = self.data[self.target_column]
            is_cat = label_vals.dtype == object or self.target_column.lower() in ["label", "class", "target"]

            if is_cat:
                self.labels = pd.Categorical(label_vals).codes.astype("int64")
                self.num_classes = len(set(self.labels))
                self.is_classification = True
            else:
                values = label_vals.values
                float_vals = [float(v) for v in values]
                unique_vals = set(float_vals)
                if len(unique_vals) <= 5 and all(v in [0, 1] for v in unique_vals):
                    self.labels = label_vals.astype(int).values
                    self.num_classes = len(set(self.labels))
                    self.is_classification = True
                else:
                    self.labels = label_vals.astype("float32").values
        except Exception:
            print("⚠️ Could not parse labels, using fallback.")
            self.labels = None

        features_df = self.data.drop(columns=[self.target_column], errors="ignore")
        features_df = features_df.apply(pd.to_numeric, errors='coerce').fillna(0)

        if normalize_input:
            mean = features_df.mean()
            std = features_df.std().replace(0, 1)
            features_df = (features_df - mean) / std
            self.feature_mean = mean
            self.feature_std = std

        if normalize_output and not self.is_classification and self.labels is not None:
            self.labels_mean = self.labels.mean()
            self.labels_std = self.labels.std() or 1
            self.labels = (self.labels - self.labels_mean) / self.labels_std

        self.features = features_df.values.astype('float32')
        self.task_type = task_type_override or ("classification" if self.is_classification else "regression")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        if self.labels is not None:
            y_type = torch.long if self.is_classification else torch.float32
            y = torch.tensor(self.labels[idx], dtype=y_type)
            return x, y.view(-1, 1) if not self.is_classification else y
        return x
