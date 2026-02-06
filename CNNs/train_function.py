import torch
import torch.nn as nn
import torch.optim as optim

from net_improve import ResizedConvFilterCNN




def train(device, output_file, train_loader, val_loader, epochs:int = 40, patience: int = 3, optimizer = "SDG", lr: float = 0.001, momentum: float = 0, weight_decay: float = 0, no_improve_break = True, kernel_size: int = 3, conv_filters = [8, 16, 32] , batch_norm = False, dropout_p = None, model_name = "best_model.pt"):

    model = ResizedConvFilterCNN(kernel_size=kernel_size, list_out_channels=conv_filters, batch_norm=batch_norm, dropout_p=dropout_p).to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer == "SDG":
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
            wd=weight_decay
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
