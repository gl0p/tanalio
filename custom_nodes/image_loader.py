import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sockets import socketio
from utils.yolo_utils import YOLODataset
from utils.coco_utils import COCODataset, coco_collate_fn
from node_registry import register_node

BASE_DATASET_DIR = os.path.join(os.getcwd(), "datasets")

@register_node(name="LoadImages", category="Loaders", tags={"run_early": True})
class LoadImages:
    outputs = [
        {"name": "train", "type": "train"},
        {"name": "val", "type": "val"},
        {"name": "test", "type": "test"},
        {"name": "sample_tensor", "type": "tensor"},
    ]

    widgets = [
        {"type": "folder_picker", "name": "folder_path", "value": ""},
        {"type": "number", "name": "batch_size", "value": 32, "options": {"min": 1, "max": 256}},
        {"type": "number", "name": "mean", "value": 0.5, "options": {"min": 0.0, "max": 1.0, "step": 0.01}},
        {"type": "number", "name": "std", "value": 0.5, "options": {"min": 0.0, "max": 1.0, "step": 0.01}},
        {"type": "number", "name": "resize_width", "value": 64},
        {"type": "number", "name": "resize_height", "value": 64},
        {"type": "label", "name": "dataset_status", "value": "Not validated"},
    ]

    def __init__(self, folder_path="", batch_size=32, resize_width=64, resize_height=64,
                 mean=0.5, std=0.5, annotation_task="instances"):
        self.folder_path = os.path.join(BASE_DATASET_DIR, folder_path)
        self.batch_size = batch_size
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.mean = mean
        self.std = std
        self.graph_node_id = None
        self.annotation_task = annotation_task
        self.task_type = None

    def _detect_format(self, path):
        if os.path.isdir(os.path.join(path, "images")) and os.path.isdir(os.path.join(path, "labels")):
            return "yolo"
        elif os.path.isdir(os.path.join(path, "annotations")):
            return "coco"
        elif os.path.isdir(os.path.join(path, "train")):
            return "folder"
        return "unknown"

    def _emit_error(self, msg):
        socketio.emit("node_error", {"node_id": self.graph_node_id})
        socketio.emit("toast", {"message": msg})
        print(msg)

    def _build_imagefolder_loader(self, transform):
        dataloaders = {}
        num_classes = None
        sample_tensor = None

        for split in ['train', 'val', 'test']:
            split_path = os.path.join(self.folder_path, split)
            if os.path.isdir(split_path):
                dataset = ImageFolder(split_path, transform=transform)
                dataloaders[split] = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                if split == 'train':
                    num_classes = len(dataset.classes)
                    try:
                        sample_tensor = next(iter(dataloaders["train"]))[0]
                        print(f"🧪 Sample tensor shape: {sample_tensor.shape}")
                    except Exception as e:
                        self._emit_error(f"❌ Error loading train sample: {str(e)}")
            else:
                print(f"⚠️ Skipped missing folder: {split_path}")

        return dataloaders, sample_tensor, num_classes, "classification"

    def _build_yolo_loader(self, transform):
        dataset = YOLODataset(
            images_dir=os.path.join(self.folder_path, "images"),
            labels_dir=os.path.join(self.folder_path, "labels"),
            transform=transform
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        sample_tensor = None
        try:
            sample_tensor, _ = next(iter(loader))
            print(f"🧪 YOLO sample tensor: {sample_tensor.shape}")
        except Exception as e:
            self._emit_error(f"❌ YOLO sample error: {str(e)}")
        return {"train": loader}, sample_tensor, 1, "detection"

    def _build_coco_loader(self, transform):
        dataloaders, sample_tensor = {}, None
        split_map = {"train": None, "val": None, "test": None}
        images_path = os.path.join(self.folder_path, "images")
        annotations_path = os.path.join(self.folder_path, "annotations")

        # find relevant annotation files
        for fname in os.listdir(annotations_path):
            if self.annotation_task in fname.lower():
                lower = fname.lower()
                if "train" in lower: split_map["train"] = fname
                if "val" in lower: split_map["val"] = fname
                if "test" in lower: split_map["test"] = fname

        for split, ann in split_map.items():
            if not ann:
                print(f"⚠️ No annotation file for {split} set")
                continue
            try:
                dataset = COCODataset(images_path, os.path.join(annotations_path, ann), transform, self.annotation_task)
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=(split == "train"), collate_fn=coco_collate_fn)
                dataloaders[split] = loader
                if split == "train":
                    sample_tensor, _ = next(iter(loader))
                    print(f"🧪 COCO train sample: {sample_tensor.shape}")
            except Exception as e:
                self._emit_error(f"❌ COCO {split} error: {str(e)}")

        if not dataloaders:
            raise Exception("❌ COCO loader failed completely.")

        return dataloaders, sample_tensor, dataset.num_classes, self.annotation_task

    def build(self):
        socketio.emit("node_active", {"node_id": self.graph_node_id})
        socketio.emit("node_clear_error", {"node_id": self.graph_node_id})

        transform = transforms.Compose([
            transforms.Resize((self.resize_height, self.resize_width)),
            transforms.ToTensor(),
            transforms.Normalize([self.mean] * 3, [self.std] * 3)
        ])

        fmt = self._detect_format(self.folder_path)
        print(f"📦 Format: {fmt}")

        dataloaders, sample_tensor, num_classes, task_type = {}, None, None, "classification"

        try:
            if fmt == "folder":
                dataloaders, sample_tensor, num_classes, task_type = self._build_imagefolder_loader(transform)
            elif fmt == "yolo":
                dataloaders, sample_tensor, num_classes, task_type = self._build_yolo_loader(transform)
            elif fmt == "coco":
                dataloaders, sample_tensor, num_classes, task_type = self._build_coco_loader(transform)
            else:
                raise Exception(f"Unknown format: {fmt}")
        except Exception as e:
            self._emit_error(f"❌ Dataset build failed: {str(e)}")

        socketio.emit("node_inactive", {"node_id": self.graph_node_id})
        self.task_type = task_type

        return {
            "train": dataloaders.get("train"),
            "val": dataloaders.get("val"),
            "test": dataloaders.get("test"),
            "sample_tensor": sample_tensor,
            "num_classes": num_classes,
            "format_type": fmt,
            "task_type": task_type
        }
