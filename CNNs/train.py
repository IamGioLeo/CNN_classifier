import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from toarchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
import os

from shallow_net import ShallowCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATE = 0.01 # To be set...
MOMENTUM = 0.9

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=transform
)

train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)

val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False
)

model = ShallowCNN.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM
)

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.maxx(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    model.eval()
    running_loss = 0.0
    corrct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            val_loss = running_loss / len(val_loader)
            val_acc = correct / total

            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                f"Val Loss {val_loss:.4f} | Val Acc: {val_acc:.3f}"
            )
