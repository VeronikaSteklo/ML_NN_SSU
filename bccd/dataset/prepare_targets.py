import yaml
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def prepare_regression_targets(data_root: str):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    df = pd.read_csv("dataset/voc_counts.csv")
    img_dir = Path(data_root) / "JPEGImages"
    df["image_path"] = df["image_id"].apply(lambda x: str(img_dir / f"{x}.jpg"))
    df[["image_path"] + cfg["CLASSES"]].to_csv("dataset/regression_targets.csv", index=False)


def create_yolo_split(data_root: str, val_ratio: float = 0.2, seed: int = 42):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    yolo_base = Path("dataset/yolo")
    flat_imgs = yolo_base / "images"
    flat_labels = yolo_base / "labels"

    if not flat_imgs.exists() or not flat_labels.exists():
        raise FileNotFoundError("Flat YOLO images/labels not found. Run voc_to_yolo first.")

    label_files = list(flat_labels.glob("*.txt"))
    ids = [f.stem for f in label_files]

    train_ids, val_ids = train_test_split(ids, test_size=val_ratio, random_state=seed)

    splits = {"train": train_ids, "val": val_ids}
    for split_name, split_ids in splits.items():
        (yolo_base / f"{split_name}/images").mkdir(parents=True, exist_ok=True)
        (yolo_base / f"{split_name}/labels").mkdir(parents=True, exist_ok=True)

        for img_id in split_ids:
            src_lbl = flat_labels / f"{img_id}.txt"
            if src_lbl.exists():
                shutil.copy2(src_lbl, yolo_base / f"{split_name}/labels/{src_lbl.name}")

            for ext in [".jpg", ".png"]:
                src_img = flat_imgs / f"{img_id}{ext}"
                if src_img.exists():
                    shutil.copy2(src_img, yolo_base / f"{split_name}/images/{src_img.name}")
                    break

    yaml_data = {
        "path": str(yolo_base.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(cfg["CLASSES"]),
        "names": cfg["CLASSES"]
    }
    with open(yolo_base / "data.yaml", "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
