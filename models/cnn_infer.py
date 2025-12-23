import torch
import torchvision.transforms as T
from torchvision.models import efficientnet_b0
from PIL import Image
import io

# --------------------------------------------------
# Load model once (module-level singleton)
# --------------------------------------------------
_model = None


def _load_model():
    global _model
    if _model is None:
        model = efficientnet_b0(weights="IMAGENET1K_V1")
        model.eval()
        _model = model
    return _model


def cnn_score(image_bytes: bytes) -> float:
    """
    Returns a heuristic AI-likelihood score based on CNN confidence.
    This is a SUPPORTING signal, not ground truth.
    """

    model = _load_model()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    # Use overconfidence as proxy
    max_prob = probs.max().item()

    # Map confidence into [0,1] AI-likelihood
    ai_score = max(0.0, max_prob - 0.6) / 0.4
    return min(ai_score, 1.0)
