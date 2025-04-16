import os
import shutil
import random
from tqdm import tqdm
from sockets import socketio
import pandas as pd
from PIL import Image

def detect_image_format(folder_path):
    if os.path.isdir(os.path.join(folder_path, "images")) and os.path.isdir(os.path.join(folder_path, "labels")):
        return "yolo"
    elif os.path.isdir(os.path.join(folder_path, "annotations")):
        return "coco"
    elif os.path.isdir(os.path.join(folder_path, "train")):
        return "folder"
    return "unknown"

def validate_or_split_dataset(base_folder, split_ratios=(0.7, 0.2, 0.1), allowed_exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    socketio.emit("toast", {"message": "Loading Dataset..."})

    format_type = detect_image_format(base_folder)
    result = {
        "status": "ok",
        "progress": 100,
        "format_type": format_type,
        "tensor_shape": None,
        "is_classification": False
    }

    if format_type == "folder":
        required_dirs = ["train", "test"]
        if all(os.path.isdir(os.path.join(base_folder, d)) for d in required_dirs):
            print("[✓] Dataset already validated.")
            return result

        all_files = [
            f for f in os.listdir(base_folder)
            if os.path.isfile(os.path.join(base_folder, f)) and f.lower().endswith(allowed_exts)
        ]

        if not all_files:
            msg = "[!] No image files found in dataset folder."
            print(msg)
            socketio.emit("toast", {"message": msg})
            return {"status": "error", "progress": 0, "message": msg}

        # Split
        random.shuffle(all_files)
        total = len(all_files)
        train_end = int(total * split_ratios[0])
        test_end = train_end + int(total * split_ratios[1])
        splits = {
            "train": all_files[:train_end],
            "test": all_files[train_end:test_end],
            "val": all_files[test_end:]
        }

        for split_name, files in tqdm(splits.items(), desc="Splitting dataset"):
            split_dir = os.path.join(base_folder, split_name)
            os.makedirs(split_dir, exist_ok=True)
            for file in files:
                src = os.path.join(base_folder, file)
                dst = os.path.join(split_dir, file)
                if os.path.exists(src):
                    shutil.move(src, dst)

        print("[✓] Dataset successfully split into train/test/val.")
        return result

    elif format_type in ["yolo", "coco"]:
        img_dir = os.path.join(base_folder, "images")
        for file in os.listdir(img_dir):
            if file.lower().endswith(allowed_exts):
                try:
                    img = Image.open(os.path.join(img_dir, file))
                    result["tensor_shape"] = (3, img.height, img.width)
                    return result
                except Exception as e:
                    return {"status": "error", "progress": 0, "message": f"❌ Image read error: {str(e)}"}

        return {"status": "error", "progress": 0, "message": "No valid images in 'images/' folder"}

    else:
        return {"status": "error", "progress": 0, "message": "Unknown dataset format"}


def validate_or_split_csv_dataset(base_folder, csv_filename=None, split_ratios=(0.7, 0.2, 0.1)):
    import csv

    socketio.emit("toast", {"message": "🧼 Cleaning and splitting CSV..."})

    base_folder = os.path.abspath(base_folder)
    csv_path = os.path.join(base_folder, csv_filename)

    if not os.path.isfile(csv_path):
        msg = f"❌ File not found: {csv_path}"
        print(msg)
        socketio.emit("toast", {"message": msg})
        return {"status": "error", "message": msg, "progress": 0}

    split_files = [os.path.join(base_folder, f"{split}.csv") for split in ["train", "val", "test"]]
    if all(os.path.isfile(f) for f in split_files):
        print("[✓] CSV already split.")
        return {
            "status": "validated",
            "progress": 100,
            "is_classification": None  # placeholder, will detect later
        }

    try:
        with open(csv_path, "r", newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        has_header = not all(cell.replace('.', '', 1).isdigit() for cell in rows[0])
        header = rows[0] if has_header else None
        data_rows = rows[1:] if has_header else rows

        # Infer if classification and get max number of features (exclude label)
        is_classification = False
        max_features_len = 0

        for row in data_rows:
            if len(row) == 0:
                continue
            if has_header and header[-1].lower() in ["label", "class", "target"]:
                is_classification = True
                features_len = len(row) - 1
            else:
                features_len = len(row)
            if features_len > max_features_len:
                max_features_len = features_len

        print(f"📏 Max feature length: {max_features_len}")

        cleaned_rows = []
        for r in data_rows:
            if is_classification:
                features = r[:-1]
                label = r[-1]
            else:
                features = r
                label = None

            # Pad features only
            features += ["0"] * (max_features_len - len(features))

            # Recombine
            cleaned = features + [label] if is_classification else features
            cleaned_rows.append(cleaned)

        # Shuffle and split
        random.shuffle(cleaned_rows)
        total = len(cleaned_rows)
        train_end = int(total * split_ratios[0])
        val_end = train_end + int(total * split_ratios[1])

        splits = {
            "train": cleaned_rows[:train_end],
            "val": cleaned_rows[train_end:val_end],
            "test": cleaned_rows[val_end:]
        }

        for split, rows in splits.items():
            split_path = os.path.join(base_folder, f"{split}.csv")
            with open(split_path, "w", newline='') as f:
                writer = csv.writer(f)
                if header:
                    writer.writerow(header)
                writer.writerows(rows)
            print(f"[✓] Wrote {split_path} ({len(rows)} rows)")

        socketio.emit("toast", {"message": "✅ CSV cleaned, padded, and split"})

        return {
            "status": "split",
            "progress": 100,
            "is_classification": is_classification
        }

    except Exception as e:
        print(f"❌ CSV error: {str(e)}")
        socketio.emit("toast", {"message": f"❌ CSV error: {str(e)}"})
        return {"status": "error", "progress": 0, "message": str(e),}


