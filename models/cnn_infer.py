import torch
import torchvision.transforms as T
from torchvision.models import efficientnet_b0
from PIL import Image
import io
import numpy as np

# -----------------------------
# Deterministic behaviour
# -----------------------------
torch.set_num_threads(1)
torch.manual_seed(0)

# -----------------------------
# Load ImageNet model
# -----------------------------
_model = efficientnet_b0(weights="IMAGENET1K_V1")
_model.eval()

_transform = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def cnn_score(image_bytes: bytes) -> float:
    """
    AI likelihood using normalized EfficientNet feature statistics.
    Higher CV => more likely AI (based on observed data).
    """

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.BICUBIC)

    x = _transform(img).unsqueeze(0)

    with torch.no_grad():
        features = _model.features(x)

    mean = features.mean().item()
    var = features.var().item()

    if mean <= 1e-6:
        print("[CNN DEBUG] Mean too small, returning 0")
        return 0.0

    cv = var / (mean * mean)

    # 🔍 Verification line (keep for now)
    print(f"[CNN DEBUG] mean={mean:.6f} | var={var:.6f} | cv={cv:.6f}")

    # -----------------------------
    # CORRECTED MAPPING (DATA-DRIVEN)
    # -----------------------------
    # Observed:
    #   Real images: cv ≈ 25–30
    #   AI images:   cv ≈ 35–45
    score = (cv - 25.0) / 20.0
    score = float(np.clip(score, 0.0, 0.85))  # safety cap

    return round(score, 3)
