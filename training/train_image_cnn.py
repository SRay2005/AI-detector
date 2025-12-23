import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = torchvision.datasets.ImageFolder("data", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = efficientnet_b0(weights="DEFAULT")
model.classifier[1] = torch.nn.Linear(1280, 1)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    for x, y in loader:
        y = y.float().unsqueeze(1)
        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "weights/image_cnn.pt")
