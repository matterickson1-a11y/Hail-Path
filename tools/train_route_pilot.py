import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# =====================
# CONFIG
# =====================

DATA_DIR = Path("training_data_route")
BATCH_SIZE = 8
EPOCHS = 8
LR = 0.0003

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# TRANSFORMS
# =====================

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =====================
# DATA
# =====================

train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

classes = train_ds.classes

print("Classes:", classes)
print("Train images:", len(train_ds))
print("Val images:", len(val_ds))

# =====================
# MODEL
# =====================

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =====================
# TRAIN LOOP
# =====================

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()

    train_acc = correct / len(train_ds)

    # VALIDATION
    model.eval()
    val_loss = 0
    val_correct = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item()
            preds = out.argmax(1)
            val_correct += (preds == y).sum().item()

    val_acc = val_correct / len(val_ds)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%}")

# =====================
# SAVE
# =====================

Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/hail_path_triage_pilot.pth")

print("\nDone.")
print("Saved model to: models/hail_path_triage_pilot.pth")