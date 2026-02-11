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
import random

from itertools import product
from csv_functions import insert_all_rows_in_csv, insert_in_csv
from train_function import train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



LEARNING_RATES = [0.001, 0.001, 0.001, 0.001, 0.001, 0.001]
MOMENTUMS = [0.0, 0.0, 0.0, 0.0, 0.0]
WEIGHT_DECAY = [0.0, 0.0, 0.0, 0.0, 0.0]
PATIENCES = [3, 3, 3, 3, 3]
KERNEL_SIZES = [3, 3, 3, 3, 3] 
CONV_FILTERS = [[8,16,32],[8,16,32],[8,16,32],[8,16,32],[8,16,32]]
OPTIMIZER = "SGD"
BATCH_NORM = False 
DROPOUT_P = None
BATCH_SIZE = 32
NUM_EPOCHS = 40
DATA_SPLIT_VERSION = "dataset_splits.pt"
DATA_AUGMENTATION = "b" #"b" for the base dataset, "m" to add the mirrored images, "a" to add augmentation
NET_VERSION = "ensamble_shallow_" + DATA_AUGMENTATION
CSV_NAME = "ensamble_final.csv"
NUM_ENSEMBLE = 5


PROJECT_ROOT = Path(__file__).resolve().parents[1]


output_dir = PROJECT_ROOT / "results/final"
output_dir.mkdir(parents=True, exist_ok=True)

file_name = f"{NET_VERSION}_results.txt" 
output_file = output_dir / file_name

curves_dir = PROJECT_ROOT / "curves/final"
curves_dir.mkdir(parents=True, exist_ok=True)

cm_dir = PROJECT_ROOT / "confusion_matrices/final"
cm_dir.mkdir(parents=True, exist_ok=True)

dataset_dir = PROJECT_ROOT / "dataset"



transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda img: torch.from_numpy(np.array(img, dtype=np.float32)).unsqueeze(0)),
    transforms.Lambda(lambda x: x - 128)
])

test_dir = dataset_dir / "resized/test"
test_set = datasets.ImageFolder(
    root=test_dir,
    transform=transform
)

base_dir = dataset_dir / "resized/train"
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

else:
    print("\r\nCreating new Train-Val-Test division")

    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size


    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    splits = {
        "train": train_set.indices,
        "val": val_set.indices
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
    mirror_dir = dataset_dir / "augmented/mirror"
    dataset_2 = datasets.ImageFolder(
        root=mirror_dir,
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

    cropped_dir = dataset_dir / "augmented/cropping"
    dataset_3 = datasets.ImageFolder(
        root=cropped_dir,
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


ensemble=[]
ensemble_id = random.randint(1,999999999)
ensemble_global_accuracy = 0
ensemble_global_total = 0
ensemble_global_correct = 0

with open(output_file, "a") as f:
    f.write("\r\n" + "-"*100)
    f.write(f"\r\nStarting new ensmeble of network training session, id = {ensemble_id}")
print("\n" + "-"*100)
print(f"Starting new ensmeble of network training session, id = {ensemble_id}")


for lr, momentum, wd, patience, kernel_size, conv_filters in zip(LEARNING_RATES, MOMENTUMS, WEIGHT_DECAY, PATIENCES, KERNEL_SIZES, CONV_FILTERS):

    model_id = random.randint(1,999999999)

    with open(output_file, "a") as f:
        f.write("\r\n" + "-"*100)
        f.write(f"\r\nTraining with LEARNING_RATE={lr}, WEIGHT_DECAY={wd}, MOMENTUM={momentum}, PATIENCE={patience}, KERNEL_SIZES={kernel_size}, CONV_FILTERS={conv_filters}")
    print("\n" + "-"*100)
    print(f"Training with LEARNING_RATE={lr}, WEIGHT_DECAY={wd}, MOMENTUM={momentum}, PATIENCE={patience}, KERNEL_SIZES={kernel_size}, CONV_FILTERS={conv_filters}")


    model, train_losses, val_losses, train_accs, val_accs, global_correct, global_total, best_val_acc, best_val_loss, best_val_acc_epoch = train(
        device=device,
        output_file=output_file,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=NUM_EPOCHS,
        patience=patience,
        optimizer=OPTIMIZER,
        lr=lr,
        momentum=momentum,
        weight_decay=wd,
        no_improve_break=False,
        kernel_size=kernel_size,
        conv_filters=conv_filters,
        batch_norm=BATCH_NORM,
        dropout_p=DROPOUT_P
    )

    model.eval()


    model_preds = []
    model_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            model_preds.append(predicted.cpu())
            model_labels.append(labels.cpu())

    model_preds = torch.cat(model_preds).numpy()
    model_labels = torch.cat(model_labels).numpy()
    model_test_accuracy = (model_preds == model_labels).mean() * 100

    insert_all_rows_in_csv(id=model_id, k_size=kernel_size,conv_fil=conv_filters, 
                lr=lr, opt=OPTIMIZER, m=momentum, wd=wd, p=patience,
                b_size=BATCH_SIZE, epochs=NUM_EPOCHS, epoch=best_val_acc_epoch, 
                b_norm=BATCH_NORM, d_out=DROPOUT_P, ens=ensemble_id, v_ls=val_losses,
                tr_ls= train_losses, v_accs=val_accs, tr_accs=train_accs,
                te_acc=model_test_accuracy, data_v=DATA_SPLIT_VERSION,
                data_aug=DATA_AUGMENTATION, csv_name=CSV_NAME)
    
    class_names =  dataset.classes

    cm = confusion_matrix(
        model_labels,
        model_preds,
    )

    global_accuracy = global_correct / global_total * 100

    ensemble_global_correct += global_correct
    ensemble_global_total += global_total

    with open(output_file, "a") as f:
        for line in [
            f"\r\nGLOBAL ACCURACY | LR={lr}, Momentum={momentum} | Accuracy: {global_accuracy:.2f}%",
            f"TEST ACCURACY | LR={lr}, Momentum={momentum} | Accuracy: {model_test_accuracy:.2f}%",
            "Confusion Matrix:",
            str(cm),
            "-"*50
        ]:
            print(line)
            f.write(line + "\n")




    class_names = dataset.classes

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
    matrix_name = f"confusion_matrix_model_id_{model_id}_ensamble_id_{ensemble_id}.png"
    fig.savefig(
        cm_dir / matrix_name,
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
        model_test_accuracy,
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
    
    fig_name = f"curves_model_id_{model_id}_ensamble_id_{ensemble_id}.png"
    fig.savefig(curves_dir / fig_name, dpi=300, bbox_inches="tight")
    plt.close(fig)

    ensemble.append(model)




all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = []

        for model in ensemble:
            outputs.append(model(images))

        outputs = torch.stack(outputs, dim=0)
        mean_outputs = outputs.mean(dim=0)

        _, predicted = torch.max(mean_outputs, 1)

        all_preds.append(predicted.cpu())
        all_labels.append(labels.cpu())

all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()
test_accuracy = (all_preds == all_labels).mean() * 100

cm = confusion_matrix(
    all_labels,
    all_preds,
)

ensemble_global_accuracy = ensemble_global_correct / ensemble_global_total *100

with open(output_file, "a") as f:
    for line in [
        f"\r\nGLOBAL ACCURACY | LRS={lr}, Momentum={momentum} | Accuracy: {ensemble_global_accuracy:.2f}%",
        f"TEST ACCURACY | LR={lr}, Momentum={momentum} | Accuracy: {test_accuracy:.2f}%",
        "Confusion Matrix:",
        str(cm),
        "-"*50
    ]:
        print(line)
        f.write(line + "\n")

insert_in_csv(id=ensemble_id, k_size=kernel_size,conv_fil=conv_filters, 
            lr=lr, opt=OPTIMIZER, m=momentum, wd=wd, p=patience,
            b_size=BATCH_SIZE, epochs=NUM_EPOCHS, epoch=best_val_acc_epoch, 
            b_norm=BATCH_NORM, d_out=DROPOUT_P, ens=NUM_ENSEMBLE, 
            te_acc=test_accuracy, data_v=DATA_SPLIT_VERSION,
            data_aug=DATA_AUGMENTATION, csv_name=CSV_NAME)

class_names = dataset.classes

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
matrix_name = f"confusion_matrix_ensamble_id_{ensemble_id}.png"
fig.savefig(
    cm_dir / matrix_name,
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)
