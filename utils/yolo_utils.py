# yolo_utils.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, img_size=(640, 640)):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform or T.ToTensor()
        self.img_size = img_size

        self.image_files = [
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Load labels
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x, y, w, h = map(float, line.strip().split())
                    # Convert YOLO format to [x1, y1, x2, y2]
                    x1 = (x - w / 2) * width
                    y1 = (y - h / 2) * height
                    x2 = (x + w / 2) * width
                    y2 = (y + h / 2) * height
                    boxes.append([class_id, x1, y1, x2, y2])

        boxes = torch.tensor(boxes) if boxes else torch.zeros((0, 5))

        if self.transform:
            image = self.transform(image)

        return image, boxes
