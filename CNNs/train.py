import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from shallow_net import ShallowCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 32
NUM_EPOCHS = [24, 40, 60]
LEARNING_RATES = [0.01, 0.001, 0.0001]
MOMENTUMS = [0.5, 0.7, 0.9]
PATIENCES = [2, 3, 5]
OUTPUT_FILE = "shallow_net_v_03_results.txt"
class_codes = [
    "00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14"
]
label_map = {
    "00": "bedroom",
    "01": "suburb",
    "02": "industrial",
    "03": "kitchen",
    "04": "living_room",
    "05": "coast",
    "06": "forest",
    "07": "highway",
    "08": "inside_city",
    "09": "mountain",
    "10": "open country",
    "11": "street",
    "12": "tall building",
    "13": "office",
    "14": "store"
}

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


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda img: torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0)),
    transforms.Lambda(lambda x: x - 128)
])


dataset = datasets.ImageFolder(
    #root="/home/leo/CNN_classifier/dataset/resized",
    root="./dataset/resized",
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

trainval_size = int(0.85 * len(dataset))
test_size = len(dataset) - trainval_size

trainval_set, test_set = torch.utils.data.random_split(
    dataset, [trainval_size, test_size]
)

val_size = int(0.15 * trainval_size)
train_size = trainval_size - val_size

## final part of data augmentation
#train_set = ConcatDataset([train_set, dataset_3])

train_set, val_set = torch.utils.data.random_split(
    trainval_set, [train_size, val_size]
)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)

#images, labels = next(iter(train_loader))
#print(images.shape)

val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=True
)


test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=True
)

sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=0.9
)


with open(OUTPUT_FILE, "a") as f:
    f.write("\r\n\nNEW TRIANING SESSION")
print("\r\n\nNEW TRIANING SESSION")
for lr in LEARNING_RATES:
    for momentum in MOMENTUMS:
        for epochs in NUM_EPOCHS:
            for patience in PATIENCES:
                with open(OUTPUT_FILE, "a") as f:
                    f.write("\r\n" + "-"*100)
                    f.write(f"\r\nTraining with LEARNING_RATE={lr}, MOMENTUM={momentum}, EPOCHS={epochs}, PATIENCE={patience}")
                print("\n" + "-"*100)
                print(f"Training with LEARNING_RATE={lr}, MOMENTUM={momentum}, EPOCHS={epochs}, PATIENCE={patience}")

                model = ShallowCNN().to(device)
                criterion = nn.CrossEntropyLoss()

                optimizer = optim.SGD(
                    model.parameters(),
                    lr=lr,
                    momentum=momentum
                )

                ## next try 
                #optimizer = optim.Adam(model.parameters(), lr=1e-3)

                train_losses, val_losses = [], []
                train_accs, val_accs = [], []
                global_correct = 0
                global_total = 0

                best_val_acc = 0
                best_val_loss = float('inf')
                epochs_no_improve = 0
                patience_trigger = False
                best_val_acc_epoch = 0
                for epoch in range(epochs):
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
                    train_acc = correct / total * 100

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

                            global_correct += (predicted == labels).sum().item()
                            global_total += labels.size(0)

                    val_loss = running_loss / len(val_loader)
                    val_acc = correct / total * 100
                    val_losses.append(val_loss)
                    val_accs.append(val_acc)
                    with open(OUTPUT_FILE, "a") as f:
                        for line in [
                            f"\r\nEpoch [{epoch+1}/{epochs}] | "
                            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                            f"Val Loss {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
                        ]:
                            f.write(line)
                    print(
                        f"Epoch [{epoch+1}/{epochs}] | "
                        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                        f"Val Loss {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
                    )

                    if not patience_trigger and val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_val_acc_epoch = epoch
                        torch.save(model.state_dict(), "best_model.pt")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= patience and not patience_trigger:
                        patience_trigger = True
                        with open(OUTPUT_FILE, "a") as f:
                            f.write("\r\nEarly stopping triggered\n")
                        print("Early stopping triggered")
                        #break

                model.load_state_dict(torch.load("best_model.pt", map_location=device))
                model.eval()

                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)

                        all_preds.append(predicted.cpu())
                        all_labels.append(labels.cpu())

                all_preds = torch.cat(all_preds).numpy()
                all_labels = torch.cat(all_labels).numpy()

                test_accuracy = (all_preds == all_labels).mean() * 100

                cm = confusion_matrix(
                    all_labels,
                    all_preds,
                    labels=np.arange(len(class_codes))
                )


                global_accuracy = global_correct / global_total * 100
                with open(OUTPUT_FILE, "a") as f:
                    for line in [
                        f"\r\nGLOBAL ACCURACY | LR={lr}, Momentum={momentum} | Accuracy: {global_accuracy:.2f}%",
                        f"TEST ACCURACY | LR={lr}, Momentum={momentum} | Accuracy: {test_accuracy:.2f}%",
                        "Confusion Matrix:",
                        str(cm),
                        "-"*50
                    ]:
                        print(line)
                        f.write(line + "\n")


                class_names = [label_map[c] for c in class_codes]
                num_classes = len(class_names)

                ticks = np.arange(num_classes)


                fig, ax = plt.subplots(figsize=(12, 10))

                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    cbar=True,
                    xticklabels=class_names,
                    yticklabels=class_names,
                    linewidths=0.5,
                    square=True,
                    ax=ax
                )

                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")

                fig.tight_layout()
                fig.savefig(
                    f"confusion_matrix_v_03_results_lr_{lr}_m_{momentum}_e_{epochs}_p_{patience}.png",
                    dpi=300,
                    bbox_inches="tight"
                )

                plt.close(fig)

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # Loss
                sns.lineplot(
                    x=np.arange(len(train_losses)),
                    y=train_losses,
                    ax=axes[0],
                    label="Train Loss",
                    marker="o"
                )
                sns.lineplot(
                    x=np.arange(len(val_losses)),
                    y=val_losses,
                    ax=axes[0],
                    label="Val Loss",
                    marker="o"
                )

                axes[0].set_title("Loss during training")
                axes[0].set_xlabel("Epoch")
                axes[0].set_ylabel("Loss")
                axes[0].legend()

                # Accuracy
                sns.lineplot(
                    x=np.arange(len(train_accs)),
                    y=train_accs,
                    ax=axes[1],
                    label="Train Accuracy",
                    marker="o"
                )
                sns.lineplot(
                    x=np.arange(len(val_accs)),
                    y=val_accs,
                    ax=axes[1],
                    label="Val Accuracy",
                    marker="o"
                )

                # Test accuracy point
                axes[1].scatter(
                    best_val_acc_epoch,
                    test_accuracy,
                    color="red",
                    s=80,
                    zorder=5,
                    label="Test Accuracy"
                )

                axes[1].set_title("Accuracy during training")
                axes[1].set_xlabel("Epoch")
                axes[1].set_ylabel("Accuracy")
                axes[1].legend()

                fig.tight_layout()

                fig.savefig(f"training_curves_v_03_results_lr_{lr}_m_{momentum}_e_{epochs}_p_{patience}.png", dpi=300, bbox_inches="tight")

                plt.close(fig)