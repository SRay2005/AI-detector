import torch
import torchvision.transforms as T
from torchvision.models import efficientnet_b0
from PIL import Image
import io

device = "cpu"

model = efficientnet_b0(weights="DEFAULT")
model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def cnn_score(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model.features(x)
        score = features.abs().mean().item()

    # normalize heuristic
    return min(score / 5.0, 1.0)
