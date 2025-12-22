import torch
import torchvision.transforms as T
from torchvision.models import efficientnet_b0
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

model = efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(1280, 1)
model.to(device)
model.eval()


def cnn_predict(image):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        out = torch.sigmoid(model(img)).item()

    return round(out, 3)
