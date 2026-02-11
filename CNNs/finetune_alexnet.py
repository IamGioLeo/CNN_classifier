import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from itertools import product
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

from train_function import finetune_alexnet
from csv_functions import insert_all_rows_in_csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


BATCH_SIZE = 32
NUM_EPOCHS = [40]
LEARNING_RATES = [0.0001]
MOMENTUMS = [0.0]
WEIGHT_DECAY = [0.001]
PATIENCES = [3]
OPTIMIZER = "Adam" #"SGD" or "Adam"
DATA_SPLIT_VERSION = "dataset_splits.pt"
DATA_AUGMENTATION = "b"  # "b" base, "m" mirror, "a" augmented
ALEXNET_VERSION = "alexnet_finetune"
ALEXNET_CSV_NAME = "alexnet_final.csv"


GRID = {
    "lr": LEARNING_RATES,
    "momentum": MOMENTUMS,
    "epochs": NUM_EPOCHS,
    "patience": PATIENCES,
    "weight_decay": WEIGHT_DECAY,
}


PROJECT_ROOT = Path(__file__).resolve().parents[1]

alexnet_output_dir = PROJECT_ROOT / "results/alexnet"
alexnet_output_dir.mkdir(parents=True, exist_ok=True)
alexnet_output_file = alexnet_output_dir / f"{ALEXNET_VERSION}_results.txt"

alexnet_curves_dir = PROJECT_ROOT / "curves/alexnet"
alexnet_curves_dir.mkdir(parents=True, exist_ok=True)

alexnet_cm_dir = PROJECT_ROOT / "confusion_matrices/alexnet"
alexnet_cm_dir.mkdir(parents=True, exist_ok=True)


dataset_dir = PROJECT_ROOT / "dataset"




train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(
        size=224,
        scale=(0.7, 1.0),
        ratio=(0.75, 1.33)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(7),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])



trainval_dataset = datasets.ImageFolder(
    root=dataset_dir / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=dataset_dir / "train",
    transform=eval_transform
)

test_set = datasets.ImageFolder(
    root=dataset_dir / "test",
    transform=eval_transform
)



if os.path.exists(DATA_SPLIT_VERSION):
    print("Loading existing Train/Val split")
    splits = torch.load(DATA_SPLIT_VERSION)
    train_set = Subset(trainval_dataset, splits["train"])
    val_set   = Subset(val_dataset, splits["val"])

else:
    print("Creating new Train/Val split")
    train_size = int(0.85 * len(trainval_dataset))
    val_size   = len(trainval_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(
        range(len(trainval_dataset)),
        [train_size, val_size]
    )
    torch.save(
        {"train": train_subset.indices, "val": val_subset.indices},
        DATA_SPLIT_VERSION
    )
    train_set = Subset(trainval_dataset, train_subset.indices)
    val_set   = Subset(val_dataset, val_subset.indices)





train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)



sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)


with open(alexnet_output_file, "a") as f:
    f.write("\n\nNEW ALEXNET FINETUNING SESSION")

print("\nNEW ALEXNET FINETUNING SESSION")


for values in product(*GRID.values()):
    params = dict(zip(GRID.keys(), values))

    print("\n" + "-" * 100)
    print(f"Training AlexNet with params: {params}")

    model, train_losses, val_losses, train_accs, val_accs, \
    best_val_acc, best_val_loss, best_val_acc_epoch = finetune_alexnet(
        device=device,
        output_file=alexnet_output_file,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=params["epochs"],
        patience=params["patience"],
        optimizer_name=OPTIMIZER,
        lr=params["lr"],
        momentum=params["momentum"],
        weight_decay=params["weight_decay"]
    )
    

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    test_accuracy = (all_preds == all_labels).mean() * 100
    cm = confusion_matrix(all_labels, all_preds)

    print(f"TEST ACCURACY: {test_accuracy:.2f}%")



    insert_all_rows_in_csv(
        k_size=None,
        conv_fil=None,
        lr=params["lr"],
        opt=OPTIMIZER,
        m=params["momentum"],
        wd=params["weight_decay"],
        p=params["patience"],
        b_size=BATCH_SIZE,
        epochs=params["epochs"],
        epoch=best_val_acc_epoch,
        b_norm=None,
        d_out=None,
        v_ls=val_losses,
        tr_ls=train_losses,
        v_accs=val_accs,
        tr_accs=train_accs,
        te_acc=test_accuracy,
        data_v=DATA_SPLIT_VERSION,
        data_aug=DATA_AUGMENTATION,
        csv_name=ALEXNET_CSV_NAME
    )




    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=trainval_dataset.classes,
        yticklabels=trainval_dataset.classes,
        square=True,
        ax=ax
    )
    ax.set_title("AlexNet Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    matrix_name = f"{ALEXNET_VERSION}_confusion_matrix_lr_{params["lr"]}_m_{params["momentum"]}_wd_{params["weight_decay"]}.png"
    fig.savefig(
        alexnet_cm_dir / matrix_name,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)



    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sns.lineplot(
        x=range(len(train_losses)), 
        y=train_losses, 
        ax=axes[0], 
        label="Train Loss",
        marker="o"
    )
    sns.lineplot(
        x=range(len(val_losses)), 
        y=val_losses, 
        ax=axes[0], 
        label="Val Loss",
        marker="o"
    )
    
    axes[0].set_title("Loss")

    sns.lineplot(
        x=range(len(train_accs)), 
        y=train_accs, ax=axes[1], 
        label="Train Acc",
        marker="o"
    )
    sns.lineplot(
        x=range(len(val_accs)), 
        y=val_accs,
        ax=axes[1], 
        label="Val Acc",
        marker="o"
    )

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
    fig_name = f"{ALEXNET_VERSION}_training_curves_lr_{params["lr"]}_m_{params["momentum"]}_wd_{params["weight_decay"]}.png"
    fig.savefig(
        alexnet_curves_dir / fig_name,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)