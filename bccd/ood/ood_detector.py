import torch
import yaml
import numpy as np
from torchvision import transforms
from PIL import Image
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

from bccd.models.models import ResNetCount


def extract_features(model, img_paths, device="cpu"):
    features = {}

    def hook(module, input, output):
        features["out"] = output.squeeze().cpu().numpy()

    model.backbone.register_forward_hook(hook)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    feats = []
    with torch.no_grad():
        for p in img_paths:
            try:
                img = transform(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
                _ = model(img)
                feats.append(features["out"])
            except Exception as e:
                print(f"Ошибка при обработке {p}: {e}")
    return np.array(feats)


def evaluate_ood(id_train_paths, id_test_paths, ood_paths, model_path, device="cpu"):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    model = ResNetCount(len(cfg["CLASSES"])).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    feats_train_id = extract_features(model, id_train_paths, device)
    feats_test_id = extract_features(model, id_test_paths, device)
    feats_ood = extract_features(model, ood_paths, device)

    n_components = min(64, len(feats_train_id))
    pca = PCA(n_components=n_components, random_state=42)

    feats_train_pca = pca.fit_transform(feats_train_id)
    feats_test_pca = pca.transform(feats_test_id)
    feats_ood_pca = pca.transform(feats_ood)

    kde = KernelDensity(kernel="gaussian", bandwidth=0.8).fit(feats_train_pca)

    scores_id = kde.score_samples(feats_test_pca)
    scores_ood = kde.score_samples(feats_ood_pca)

    novelty_id = -scores_id
    novelty_ood = -scores_ood

    y_true = np.concatenate([np.zeros(len(novelty_id)), np.ones(len(novelty_ood))])
    y_scores = np.concatenate([novelty_id, novelty_ood])

    auc = roc_auc_score(y_true, y_scores)

    threshold_95 = np.percentile(novelty_ood, 5)
    fpr95 = np.mean(novelty_id > threshold_95)

    return auc, fpr95