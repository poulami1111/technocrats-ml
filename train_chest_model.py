import os
import torch
from torchvision import datasets, models, transforms
from torch import nn, optim
from torch.utils.data import DataLoader

# === Paths ===
DATA_DIR = "chest-dataset-enhanced"
MODEL_SAVE_PATH = "chest_disease_model.pth"

# === Training Config ===
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 0.001
IMAGE_SIZE = 224

# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Training on device: {device}")

# === Transforms ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Since we converted grayscale to RGB
])

# === Dataset & Dataloader ===
print("📁 Loading dataset...")
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
class_names = dataset.classes
print(f"📁 Classes: {class_names}")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print("✅ Dataloader ready. Starting training loop...")

# === Model ===
model = models.resnet18(weights='IMAGENET1K_V1')  # pretrained
model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Custom classification head
model = model.to(device)

# === Loss & Optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# === Training Loop ===
print("🚀 Starting training...\n")
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    print(f"✅ Epoch {epoch+1}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.2f}%\n")

# === Save the model ===
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"💾 Model saved to {MODEL_SAVE_PATH}")
