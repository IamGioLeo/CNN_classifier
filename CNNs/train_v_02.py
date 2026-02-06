import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pathlib import Path

from itertools import product
from net_improve import ResizedConvFilterCNN
from csv_functions import insert_in_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 48
NUM_EPOCHS = [40]
LEARNING_RATES = [0.0001]
MOMENTUMS = [0.9]
PATIENCES = [3]
KERNEL_SIZES = [3,5,7]
CONV_FILTERS = [[16, 32, 64, 128],[8,16,32,64],[16,32,64]]
BATCH_NORM = False
DROPOUT = None
DATA_SPLIT_VERSION = "dataset_splits.pt"
DATA_AUGMENTATION = "a" #"b" for the base dataset, "m" to add the mirrored images, "a" to add augmentation
NET_VERSION = "conv_filters_v_01_" + DATA_AUGMENTATION
CSV_NAME = "csv_v1.csv"


GRID = {
    "lr": LEARNING_RATES,
    "momentum": MOMENTUMS,
    "epochs": NUM_EPOCHS,
    "patience": PATIENCES,
    "kernel_size": KERNEL_SIZES,
    "conv_filters": CONV_FILTERS,
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


file_name = f"results/{NET_VERSION}_results.txt" #conv_filters_v_01_results.txt"
output_file = PROJECT_ROOT / file_name

output_dir = PROJECT_ROOT / "results"
output_dir.mkdir(parents=True, exist_ok=True)

curves_dir = PROJECT_ROOT / "curves"
curves_dir.mkdir(parents=True, exist_ok=True)

cm_dir = PROJECT_ROOT / "confusion_matrices"
cm_dir.mkdir(parents=True, exist_ok=True)

dataset_dir = PROJECT_ROOT / "dataset"



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

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda img: torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0)),
    transforms.Lambda(lambda x: x - 128)
])


base_dir = dataset_dir / "resized"
dataset = datasets.ImageFolder(
    root=base_dir,
    transform=transform
)

# this can be used to keep the same train, val, test division, 
# is needed to keep a proper separation between train and other sets when augmenting 
splits_path = DATA_SPLIT_VERSION

if os.path.exists(splits_path):
    print("\r\nLoading old Train-Val-Test division")

    splits = torch.load(splits_path)

    train_set = Subset(dataset, splits["train"])
    val_set   = Subset(dataset, splits["val"])
    test_set  = Subset(dataset, splits["test"])

else:
    print("\r\nCreating new Train-Val-Test division")

    trainval_size = int(0.85 * len(dataset))
    test_size = len(dataset) - trainval_size

    val_size = int(0.15 * trainval_size)
    train_size = trainval_size - val_size

    train_set, test_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, test_size, val_size]
    )

    splits = {
        "train": train_set.indices,
        "val": val_set.indices,
        "test": test_set.indices
    }

    torch.save(splits, DATA_SPLIT_VERSION)


if DATA_AUGMENTATION == "m" or DATA_AUGMENTATION == "a":
    print("Adding mirrored images to training set")

    train_filenames = set()

    for idx in train_set.indices:
        path, _ = dataset.samples[idx]
        filename = os.path.basename(path)
        train_filenames.add(filename)

    # mirrror subset for training 
    dataset_2 = datasets.ImageFolder(
        root="/home/leo/CNN_classifier/dataset/augmented/mirror",
        transform=transform
    )

    mirror_indices = [
        i for i, (path, _) in enumerate(dataset_2.samples)
        if os.path.basename(path) in train_filenames
    ]

    mirror_train_set = Subset(dataset_2, mirror_indices)

    train_set = ConcatDataset([train_set, mirror_train_set])


if DATA_AUGMENTATION == "a":
    # agmented subset for training
    print("Adding cropped images to training set")

    dataset_3 = datasets.ImageFolder(
        root="/home/leo/CNN_classifier/dataset/augmented/cropping",
        transform=transform
    )

    def is_cropped_version(crop_name, original_names):
        for name in original_names:
            stem = os.path.splitext(name)[0]
            if crop_name.endswith(f"_{stem}"):
                return True
        return False

    crop_indices = [
        i for i, (path, _) in enumerate(dataset_3.samples)
        if is_cropped_version(os.path.basename(path), train_filenames)
    ]

    crop_train_set = Subset(dataset_3, crop_indices)
    train_set = ConcatDataset([train_set, crop_train_set])




train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False
)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False
)


sns.set_theme(
    style="whitegrid",
    context="talk",
    font_scale=0.9
)


with open(output_file, "a") as f:
    f.write("\r\n\nNEW TRIANING SESSION")
print("\r\n\nNEW TRIANING SESSION")


for values in product(*GRID.values()):
    params = dict(zip(GRID.keys(), values))

    lr = params["lr"]
    momentum = params["momentum"]
    epochs = params["epochs"]
    patience = params["patience"]
    kernel_size = params["kernel_size"]
    conv_filters = params["conv_filters"]

    with open(output_file, "a") as f:
        f.write("\r\n" + "-"*100)
        f.write(f"\r\nTraining with LEARNING_RATE={lr}, MOMENTUM={momentum}, EPOCHS={epochs}, PATIENCE={patience}, KERNEL_SIZES={kernel_size}, CONV_FILTERS={conv_filters}")
    print("\n" + "-"*100)
    print(f"Training with LEARNING_RATE={lr}, MOMENTUM={momentum}, EPOCHS={epochs}, PATIENCE={patience}, KERNEL_SIZES={kernel_size}, CONV_FILTERS={conv_filters}")
    
    model = ResizedConvFilterCNN(kernel_size=kernel_size, list_out_channels=conv_filters, batch_norm=BATCH_NORM, dropout_p=DROPOUT).to(device)
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
            torch.save(model.state_dict(), "best_model.pt")

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
            #break

        insert_in_csv(k_size=kernel_size,conv_fil=conv_filters, 
                      lr=lr, opt="SDG", m=momentum, wd=None, p=patience,
                      b_size=BATCH_SIZE, epochs=epochs, epoch=epoch, b_norm=BATCH_NORM, 
                      d_out=DROPOUT, ens=None, v_l=val_loss, tr_l=train_loss,
                      v_acc=val_acc, tr_acc=train_acc, data_v=DATA_SPLIT_VERSION,
                      data_aug=DATA_AUGMENTATION, csv_name=CSV_NAME)

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

    with open(output_file, "a") as f:
        for line in [
            f"\r\nGLOBAL ACCURACY | LR={lr}, Momentum={momentum} | Accuracy: {global_accuracy:.2f}%",
            f"TEST ACCURACY | LR={lr}, Momentum={momentum} | Accuracy: {test_accuracy:.2f}%",
            "Confusion Matrix:",
            str(cm),
            "-"*50
        ]:
            print(line)
            f.write(line + "\n")

    insert_in_csv(k_size=kernel_size,conv_fil=conv_filters, 
                  lr=lr, opt="SDG", m=momentum, wd=None, p=patience,
                  b_size=BATCH_SIZE, epochs=epochs, epoch=best_val_acc_epoch, 
                  b_norm=BATCH_NORM, d_out=DROPOUT, ens=None, 
                  te_acc=test_accuracy, data_v=DATA_SPLIT_VERSION,
                  data_aug=DATA_AUGMENTATION, csv_name=CSV_NAME)
            
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
    matrix_name = f"confusion_matrices/{NET_VERSION}_confusion_matrix_lr_{lr}_m_{momentum}_e_{epochs}_p_{patience}_ks_{kernel_size}_cf_{conv_filters}.png"
    fig.savefig(
        PROJECT_ROOT / matrix_name,
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
    
    fig_name = f"curves/{NET_VERSION}_training_curves_lr_{lr}_m_{momentum}_e_{epochs}_p_{patience}_ks_{kernel_size}_cf_{conv_filters}.png"
    fig.savefig(PROJECT_ROOT / fig_name, dpi=300, bbox_inches="tight")
    plt.close(fig)