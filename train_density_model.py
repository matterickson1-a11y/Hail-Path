import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

DATA_DIR = "data/density_dataset"
BATCH_SIZE = 16
NUM_EPOCHS = 5
LR = 1e-4
IMG_SIZE = 224
MODEL_SAVE_PATH = "models/density_classifier.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=val_transform
)

train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

train_labels = [label for _, label in train_dataset.samples]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = torch.tensor(class_weights,dtype=torch.float).to(DEVICE)

print("Class weights:")
for i,name in enumerate(class_names):
    print(name,class_weights[i].item())

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model.fc = nn.Linear(model.fc.in_features,num_classes)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(),lr=LR)

best_acc = 0

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0

    for images,labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)

    model.eval()

    val_loss = 0
    correct = 0

    with torch.no_grad():

        for images,labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs,labels)

            val_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs,1)

            correct += (preds == labels).sum().item()

    val_loss /= len(val_dataset)

    val_acc = correct / len(val_dataset)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.2%}")

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save({
            "model_state_dict":model.state_dict(),
            "class_names":class_names
        },MODEL_SAVE_PATH)

        print("Saved best model")

print("Training complete")
print("Best validation accuracy:",best_acc)