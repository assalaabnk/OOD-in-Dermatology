import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.model_selection import KFold
import torchmetrics
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd


def train_and_evaluate(
    ae, criterion, optimizer, dataset, epochs=22, batch_size=64, num_folds=4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    auc_scores = []
    f1_scores = []

    splits = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(splits.split(dataset)):
        nro_fold = fold_idx + 1
        print("Fold {}".format(nro_fold))

        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]

        # Create data loaders for training and validation data
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        running_loss = 0.0

        # Training loop
        for epoch in range(epochs):
            for i, (inputs, _) in enumerate(train_loader, 0):
                inputs = inputs.to(device)

                # Forward Pass
                encoded, outputs = ae(inputs)
                loss = criterion(outputs, inputs)

                # Backward Pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 50 == 49:  # Print every 50 mini-batches
                    print(
                        f"[Fold {nro_fold}, Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 50:.3f}"
                    )
                    running_loss = 0.0

        ae.eval()
        true_labels = []
        predicted_labels = []

        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                encoded, outputs = ae(inputs)
                predicted = (outputs > 0.5).float()

                true_labels.extend(inputs.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())

        # Calculate AUC and F1 score for this fold
        aucroc = torchmetrics.AUROC(task="binary")
        auc = aucroc(torch.tensor(true_labels), torch.tensor(predicted_labels))
        auc_scores.append(auc.item())

        metric = torchmetrics.classification.BinaryF1Score()
        f1 = metric(torch.tensor(true_labels), torch.tensor(predicted_labels))
        f1_scores.append(f1.item())

    # Calculate the mean and standard deviation of AUC and F1 scores
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)

    print(f"Mean AUC: {mean_auc:.4f}, Std AUC: {std_auc:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}, Std F1 Score: {std_f1:.4f}")
