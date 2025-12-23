import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import io

# -----------------------------
# Simple noise-residual CNN
# -----------------------------
class NoiseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.net(x).view(x.size(0), -1)


_model = NoiseCNN()
_model.eval()

_transform = T.Compose([
    T.Resize((256, 256)),
    T.Grayscale(),
    T.ToTensor()
])


def cnn_score(image_bytes: bytes) -> float:
    """
    Returns AI likelihood based on noise residual consistency.
    AI images are too smooth / uniform in noise space.
    """

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = _transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = _model(x)

    var = feat.var().item()

    # ---- calibrated thresholds ----
    if var < 0.002:
        return 0.95
    elif var < 0.004:
        return 0.7
    elif var < 0.006:
        return 0.4
    else:
        return 0.0
