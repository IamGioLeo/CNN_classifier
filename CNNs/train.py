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


from shallow_net import ShallowCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 32
NUM_EPOCHS = 40
LEARNING_RATES = [0.01, 0.001, 0.0001] # To be set...
MOMENTUMS = [0.5, 0.7, 0.9]
PATIENCE = 5
OUTPUT_FILE = "shallow_net_v_01_results.txt"

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
with open(OUTPUT_FILE, "a") as f:
    f.write("\r\n\nNEW TRIANING SESSION")
print("\r\n\nNEW TRIANING SESSION")
for lr in LEARNING_RATES:
    for momentum in MOMENTUMS:
        with open(OUTPUT_FILE, "a") as f:
            f.write("\r\n" + "-"*100)
            f.write(f"\r\nTraining with LEARNING_RATE={lr}, MOMENTUM={momentum}")
        print("\n" + "-"*100)
        print(f"Training with LEARNING_RATE={lr}, MOMENTUM={momentum}")

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
                    f"\r\nEpoch [{epoch+1}/{NUM_EPOCHS}] | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Loss {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
                ]:
                    f.write(line)
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
            )
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_model.pt")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= PATIENCE:
                with open(OUTPUT_FILE, "a") as f:
                    f.write("\r\nEarly stopping triggered\n")
                print("Early stopping triggered")
                break
                         
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

        cm = confusion_matrix(all_labels, all_preds)


        global_accuracy = global_correct / global_total * 100
        with open(OUTPUT_FILE, "a") as f:
            for line in [
                f"GLOBAL ACCURACY | LR={lr}, Momentum={momentum} | Accuracy: {global_accuracy:.2f}%",
                f"TEST ACCURACY | LR={lr}, Momentum={momentum} | Accuracy: {test_accuracy:.2f}%",
                "Confusion Matrix:",
                str(cm),
                "-"*50
            ]:
                print(line)
                f.write(line + "\n")
        #plt.figure(figsize=(6, 6))
        #plt.imshow(cm)
        #plt.title("Confusion Matrix")
        #plt.xlabel("Predicted")
        #plt.ylabel("True")
        #plt.colorbar()
        #plt.show()
        

# plt.figure(figsize=(10, 4))
# 
# # Loss
# plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Val Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss during training')
# plt.legend()
# plt.grid(True)
# 
# # Accuracy
# plt.subplot(1, 2, 2)
# plt.plot(train_accs, label='Train Accuracy')
# plt.plot(val_accs, label='Val Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy during training')
# plt.legend()
# plt.grid(True)
# 
# plt.tight_layout()
# plt.show()