import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

from skimage import data
from bccd.ood.ood_detector import evaluate_ood
from bccd.ood.generate_corruptions import apply_corruptions


def prepare_real_microscopy_ood(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(list(out_dir.glob("*.jpg"))) > 0:
        return

    samples = {
        "coins": data.coins(),
        "gravel": data.gravel(),
        "brick": data.brick(),
        "grass": data.grass()
    }

    for name, img_gray in samples.items():
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_res = cv2.resize(img_rgb, (640, 640))

        cv2.imwrite(str(out_dir / f"local_ood_{name}.jpg"), img_res)


def run_ood_pipeline():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    model_path = "outputs/models/regression_resnet.pth"

    df = pd.read_csv("dataset/regression_targets.csv")
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=cfg["SEED"])
    id_train_paths = train_df["image_path"].tolist()
    id_test_paths = test_df["image_path"].tolist()

    synth_dir = Path(cfg["OUTPUT_DIR"]) / "ood_corrupted"
    if not synth_dir.exists() or not list(synth_dir.glob("*.jpg")):
        apply_corruptions(cfg["DATA_ROOT"], str(synth_dir))

    real_ood_dir = Path("data/ood_real_micro")
    prepare_real_microscopy_ood(real_ood_dir)
    real_ood_paths = list(real_ood_dir.glob("*.jpg"))

    results = {}

    for corr in ["blur", "noise"]:
        paths = list(synth_dir.glob(f"{corr}_*.jpg"))
        if paths:
            auc, fpr95 = evaluate_ood(id_train_paths, id_test_paths, paths, model_path)
            results[corr] = {"AUROC": auc, "FPR95": fpr95}

    if real_ood_paths:
        auc, fpr95 = evaluate_ood(id_train_paths, id_test_paths, real_ood_paths, model_path)
        results["real_microscopy"] = {"AUROC": auc, "FPR95": fpr95}

    res_df = pd.DataFrame(results).T
    res_df.to_csv("outputs/ood_metrics.csv")

    plt.figure(figsize=(10, 5))
    sns.barplot(x=res_df.index, y=res_df["AUROC"], palette="magma")
    plt.title("OOD Detection: Blood vs Other Biology")
    plt.ylabel("AUROC score")
    plt.ylim(0.5, 1.02)
    plt.tight_layout()
    plt.savefig("outputs/ood_performance.png")
    plt.show()

    return results
