import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import AlexNet_Weights
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from net_improve import ResizedConvFilterCNN




def train(device, output_file, train_loader, val_loader, epochs:int = 40, patience: int = 3, optimizer = "SGD", lr: float = 0.001, momentum: float = 0, weight_decay: float = 0, no_improve_break = True, kernel_size: int = 3, conv_filters = [8, 16, 32] , batch_norm = False, dropout_p = None, model_name = "best_model.pt"):

    model = ResizedConvFilterCNN(kernel_size=kernel_size, list_out_channels=conv_filters, batch_norm=batch_norm, dropout_p=dropout_p).to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

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

        with open(output_file, "a") as f:
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
            torch.save(model.state_dict(), model_name)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and not patience_trigger:
            patience_trigger = True
            with open(output_file, "a") as f:
                f.write("\r\nEarly stopping triggered\n")
            print("Early stopping triggered")
            if no_improve_break:
                break

        if patience_trigger and best_val_acc_epoch < epoch - 4:
            break

    model.load_state_dict(torch.load(model_name, map_location=device))
     
    return (
        model,
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        global_correct,
        global_total,
        best_val_acc,
        best_val_loss,
        best_val_acc_epoch
    )






def finetune_alexnet(device, output_file, train_loader, val_loader, epochs: int = 40, patience: int = 3, optimizer_name: str = "SGD", lr: float = 1e-3, momentum: float = 0.9, weight_decay: float = 0, no_improve_break: bool = True, model_name: str = "best_alexnet.pt"):
    weights = AlexNet_Weights.DEFAULT
    model = models.alexnet(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(4096, 15)


    for param in model.classifier[6].parameters():
        param.requires_grad = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "SGD":
        optimizer = optim.SGD(
            model.classifier[6].parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        optimizer = optim.Adam(
            model.classifier[6].parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
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
        val_loss = running_loss / len(val_loader)
        val_acc = correct / total * 100
        val_losses.append(val_loss)
        val_accs.append(val_acc)


        log_line = (
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        print(log_line)
        with open(output_file, "a") as f:
            f.write("\n" + log_line)


        if not patience_trigger and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_acc_epoch = epoch
            torch.save(model.state_dict(), model_name)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and not patience_trigger:
            patience_trigger = True
            print("Early stopping triggered")
            if no_improve_break:
                break

        if patience_trigger and best_val_acc_epoch < epoch - 4:
            break

    model.load_state_dict(torch.load(model_name, map_location=device))

    return (
        model,
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        best_val_acc,
        best_val_loss,
        best_val_acc_epoch
    )



def extract_features(device, dataloader, model):
    model.eval()
    features_list, labels_list = [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)

            x = model.features(images)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            x = model.classifier[:-1](x)

            features_list.append(x.cpu().numpy())
            labels_list.append(labels.numpy())

    features_arr = np.concatenate(features_list, axis=0)
    labels_arr = np.concatenate(labels_list, axis=0)
    return features_arr, labels_arr



def train_multiclass_svm(features_train, labels_train, features_val=None, labels_val=None, C=1.0):
    
    scaler = StandardScaler()
    features_train = scaler.fit_transform(features_train)
    if features_val is not None:
        features_val = scaler.transform(features_val)

    svm_model = SVC(kernel="linear", C=C, decision_function_shape="ovr")
    svm_model.fit(features_train, labels_train)

    if features_val is not None:
        acc = svm_model.score(features_val, labels_val) * 100
        print(f"Validation Accuracy: {acc:.2f}%")

    return svm_model, scaler
