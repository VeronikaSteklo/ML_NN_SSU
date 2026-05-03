import cv2
import yaml
from pathlib import Path
from tqdm import tqdm
import albumentations as A


def apply_corruptions(data_root: str, out_dir: str):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    img_dir = Path(data_root) / "JPEGImages"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    transforms_dict = {
        "blur": A.Compose([A.GaussianBlur(blur_limit=(3, 9), p=1.0)]),
        "noise": A.Compose([A.GaussNoise(var_limit=(30, 100), p=1.0)]),
        "jpeg": A.Compose([A.ImageCompression(quality_range=(20, 50), p=1.0)]),
        "contrast": A.Compose([A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.5, p=1.0)]),
        "defocus": A.Compose([A.Defocus(radius=(2, 5), alias_blur=(0.1, 0.3), p=1.0)]),
        "motion": A.Compose([A.MotionBlur(blur_limit=7, p=1.0)])
    }

    for img_path in tqdm(img_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for name, tr in transforms_dict.items():
            aug = tr(image=img_rgb)["image"]
            aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out / f"{name}_{img_path.name}"), aug_bgr)

    print(f"✅ Corrupted images saved to {out}")
