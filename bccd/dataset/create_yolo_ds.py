import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
import shutil  # ← добавили


def voc_to_yolo(data_root: str):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    ann_dir = Path(data_root) / "Annotations"
    img_dir = Path(data_root) / "JPEGImages"
    yolo_labels = Path("dataset/yolo/labels")
    yolo_imgs = Path("dataset/yolo/images")

    yolo_labels.mkdir(parents=True, exist_ok=True)
    yolo_imgs.mkdir(parents=True, exist_ok=True)

    cls_map = {name: i for i, name in enumerate(cfg["CLASSES"])}

    for xml in ann_dir.glob("*.xml"):
        tree = ET.parse(xml)
        root = tree.getroot()
        w = int(root.find("size/width").text)
        h = int(root.find("size/height").text)
        img_id = root.find("filename").text

        lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in cls_map:
                continue
            box = obj.find("bndbox")
            x1, y1, x2, y2 = map(float, [box.find(t).text for t in ["xmin", "ymin", "xmax", "ymax"]])
            xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
            bw, bh = x2 - x1, y2 - y1
            lines.append(f"{cls_map[cls_name]} {xc / w:.6f} {yc / h:.6f} {bw / w:.6f} {bh / h:.6f}")

        if lines:
            label_name = Path(img_id).stem + ".txt"
            (yolo_labels / label_name).write_text("\n".join(lines) + "\n")
            shutil.copy2(img_dir / img_id, yolo_imgs / img_id)
