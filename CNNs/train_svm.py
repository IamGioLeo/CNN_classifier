import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torchvision import models
from torchvision.models import AlexNet_Weights
import numpy as np
from sklearn.metrics import confusion_matrix

from train_function import extract_features, train_multiclass_svm
from csv_functions import insert_all_rows_in_csv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


BATCH_SIZE = 32
DATA_SPLIT_VERSION = "dataset_splits.pt"
SVM_VERSION = "svm"
SVM_CSV_NAME = "svm_final.csv"
SVM_C = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


PROJECT_ROOT = Path(__file__).resolve().parents[1]



svm_output_dir = PROJECT_ROOT / "results/svm"
svm_output_dir.mkdir(parents=True, exist_ok=True)
svm_output_file = svm_output_dir / f"{SVM_VERSION}_results.txt"

svm_cm_dir = PROJECT_ROOT / "confusion_matrices/svm"
svm_cm_dir.mkdir(parents=True, exist_ok=True)

dataset_dir = PROJECT_ROOT / "dataset"





transform = transforms.Compose([
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
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root=dataset_dir / "train",
    transform=transform
)

test_set = datasets.ImageFolder(
    root=dataset_dir / "test",
    transform=transform
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



weights = AlexNet_Weights.DEFAULT
model = models.alexnet(weights=weights)

for param in model.parameters():
    param.requires_grad = False

model = model.to(device)

print("\nExtracting features from train, val, and test sets...")
features_train, labels_train = extract_features(device, train_loader, model)
features_val, labels_val = extract_features(device, val_loader, model)
features_test, labels_test = extract_features(device, test_loader, model)


sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)

best_c = None
best_val_acc = -1.0
svm_results = {}

for c in SVM_C:

    print("\n" + "-" * 100)
    print(f"Training SVM with c: {c}")
    
    print("\nTraining multiclass linear SVM...")
    svm_model, scaler = train_multiclass_svm(features_train, labels_train, features_val, labels_val, C=c)

    val_features_scaled = scaler.transform(features_val)
    val_acc = svm_model.score(val_features_scaled, labels_val) * 100

    svm_results[c] = val_acc
    print(f"SVM Val Accuracy: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_c = c

print(f"\nRetraining SVM with best C = {best_c} on train+val")


features_trainval = np.vstack([features_train, features_val])
labels_trainval   = np.hstack([labels_train, labels_val])


svm_model, scaler = train_multiclass_svm(features_trainval,labels_trainval,C=best_c)


features_test_scaled = scaler.transform(features_test)
test_acc_svm = svm_model.score(features_test_scaled, labels_test) * 100

print(f"SVM FINAL Test Accuracy (C={best_c}): {test_acc_svm:.2f}%")


insert_all_rows_in_csv(
    k_size=None,
    conv_fil=None,
    lr=None,
    opt="LinearSVM",
    m=None,
    wd=None,
    p=None,
    b_size=BATCH_SIZE,
    epochs=1,
    epoch=0,
    b_norm=None,
    d_out=features_train.shape[1],  # 4096
    v_ls=list(svm_results.values()),
    tr_ls=[None],
    v_accs=list(svm_results.values()),
    tr_accs=[None],
    te_acc=test_acc_svm,
    data_v=DATA_SPLIT_VERSION,
    data_aug="b",
    csv_name=SVM_CSV_NAME
)


cm_svm = confusion_matrix(
    labels_test,
    svm_model.predict(features_test_scaled)
)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    cm_svm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=True,
    xticklabels=trainval_dataset.classes,
    yticklabels=trainval_dataset.classes,
    square=True,
    ax=ax
)

ax.set_title("SVM Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

fig.savefig(
    svm_cm_dir / f"{SVM_VERSION}_bestC_{best_c}_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)
