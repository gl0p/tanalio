import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms.functional as TF

class COCODataset(Dataset):
    def __init__(self, images_dir, annotation_file, transform=None, task_type="instances"):
        self.images_dir = images_dir
        self.coco = COCO(annotation_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.task_type = task_type

        # ✅ Only compute num_classes for relevant tasks
        if task_type == "instances" or task_type == "classification":
            self.categories = self.coco.loadCats(self.coco.getCatIds())
            self.num_classes = len(self.categories)
        else:
            self.num_classes = 1  # fallback default

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        if self.task_type == "instances":
            boxes = []
            labels = []
            for ann in anns:
                if ann.get("iscrowd", 0) == 1: continue
                x, y, w, h = ann["bbox"]
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "image_id": torch.tensor([img_id]),
            }
            return image, target


        elif self.task_type == "captions":
            captions = [ann["caption"] for ann in anns]
            text = captions[0] if captions else ""

            # 👇 TEMP simple encoding to char-level indices (replace w/ tokenizer later)
            tokens = [ord(c) for c in text][:256]  # truncate or pad
            target = torch.tensor(tokens, dtype=torch.long)
            return image, target


        elif self.task_type == "keypoints":
            keypoints = [ann["keypoints"] for ann in anns if "keypoints" in ann]
            return image, keypoints[0] if keypoints else []

        # Default fallback for classification-style use
        elif self.task_type == "classification":
            for ann in anns:
                if ann.get("iscrowd", 0) == 1: continue
                return image, torch.tensor(ann["category_id"], dtype=torch.long)
            return image, torch.tensor(0, dtype=torch.long)


def coco_collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)

    if isinstance(targets[0], torch.Tensor):
        try:
            targets = torch.stack(targets)
        except:
            pass  # fallback if they’re ragged
    return images, targets


