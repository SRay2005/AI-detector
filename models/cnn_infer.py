import torch
import torchvision.transforms as T
from torchvision.models import efficientnet_b0
from PIL import Image
import io
import numpy as np

# ❌ NO deterministic settings here

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


def cnn_score_once(image_bytes: bytes) -> float:
    """
    Single stochastic CNN pass (CP1 style).
    """

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Slight stochastic resizing (key!)
    jitter = np.random.randint(-4, 5)
    size = 224 + jitter
    img = img.resize((size, size), Image.BICUBIC)

    x = _transform(img).unsqueeze(0)

    with torch.no_grad():
        features = _model.features(x)

    mean = features.mean().item()
    var = features.var().item()

    if mean <= 1e-6:
        return 0.0

    cv = var / (mean * mean)

    # CP1-style loose mapping
    score = (cv - 25.0) / 20.0
    return float(np.clip(score, 0.0, 1.0))


def cnn_score(image_bytes: bytes, runs: int = 7) -> dict:
    """
    Multi-run CNN score.
    Returns mean + std.
    """

    scores = [cnn_score_once(image_bytes) for _ in range(runs)]

    return {
        "mean": round(float(np.mean(scores)), 3),
        "std": round(float(np.std(scores)), 3)
    }
