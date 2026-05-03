import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path


def parse_voc(data_root: str) -> pd.DataFrame:
    ann_dir = Path(data_root) / "Annotations"
    records = []
    for xml_path in ann_dir.glob("*.xml"):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_id = root.find("filename").text.replace(".jpg", "")
        counts = {"WBC": 0, "RBC": 0, "Platelets": 0}
        for obj in root.findall("object"):
            cls = obj.find("name").text
            if cls in counts:
                counts[cls] += 1
        records.append({"image_id": img_id, **counts})
    return pd.DataFrame(records)
