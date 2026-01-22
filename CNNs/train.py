import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
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


## this is for not correct size images
#transform = transforms.Compose([
#    transforms.Grayscale(num_output_channels=1),
#    transforms.Resize((64, 64)),
#    transforms.ToTensor()
#])


## this is for correct 64 * 64 size images --> for some reason it doesn't work properly so i had to change it
#
#transform = transforms.Compose([
#    transforms.Grayscale(num_output_channels=1),
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.5], std=[0.5])
#])

# looks like the problem with the ToTensor() method is that puts the valuse of the array in the range [0,1]
# with the initialization of weights parameters the professor told us this is distruptive:
# weights start with low values * low values of the ToTensor method make the network almost inactive

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda img: torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0)),
    transforms.Lambda(lambda x: x - 128)
])


dataset = datasets.ImageFolder(
    root="/home/leo/CNN_classifier/dataset/resized",
    transform=transform
)

## this will be used later to have a proper augmentation of the dataset
#
#dataset_2 = datasets.ImageFolder(
#    root="/home/leo/CNN_classifier/dataset/augmented/mirror",
#    transform=transform
#)
#
#dataset_3 = datasets.ImageFolder(
#    root="/home/leo/CNN_classifier/dataset/augmented/cropping",
#    transform=transform
#)
#
#dataset = ConcatDataset([dataset, dataset_2])

train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

## final part of data augmentation
#train_set = ConcatDataset([train_set, dataset_3])

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)

#images, labels = next(iter(train_loader))
#print(images.shape)

val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False
)

model = ShallowCNN().to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM
)

## next try 
#optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    train_losses.append(train_loss)
    train_accs.append(train_acc)

    model.eval()
    running_loss = 0.0
    correct = 0
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
