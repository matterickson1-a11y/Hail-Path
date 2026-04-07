import os
import copy
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

DATASET_DIR = "dataset"
MODEL_PATH = "models/hail_path_triage.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 0.0005
VAL_SPLIT = 0.2
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.08, contrast=0.08),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

full_dataset = ImageFolder(DATASET_DIR)
class_names = full_dataset.classes
print("Classes found:", class_names)

if len(class_names) != 3:
    raise ValueError("Expected classes: green_pdr, red_conventional, yellow_review")

dataset_size = len(full_dataset)
val_size = max(1, int(dataset_size * VAL_SPLIT))
train_size = dataset_size - val_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_counts = [0] * len(class_names)
for _, label in full_dataset.samples:
    class_counts[label] += 1

print("Class counts:")
for i, name in enumerate(class_names):
    print(f"  {name}: {class_counts[i]}")

total_samples = sum(class_counts)
class_weights = [total_samples / max(count, 1) for count in class_counts]
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

print("Class weights:")
for i, name in enumerate(class_names):
    print(f"  {name}: {class_weights[i]:.4f}")

model = torchvision.models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, 3)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")
best_state = None

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_train_loss = train_loss / max(len(train_loader), 1)
    avg_val_loss = val_loss / max(len(val_loader), 1)
    val_acc = correct / max(total, 1)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state = copy.deepcopy(model.state_dict())

if best_state is None:
    best_state = model.state_dict()

os.makedirs("models", exist_ok=True)
torch.save({
    "model_state_dict": best_state,
    "class_names": class_names
}, MODEL_PATH)

print(f"Saved BEST triage model to {MODEL_PATH}")