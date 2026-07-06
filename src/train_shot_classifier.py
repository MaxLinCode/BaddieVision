import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import GroupShuffleSplit

try:
    from .court_features import CalibrationRegistry
except ImportError:
    from court_features import CalibrationRegistry

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "clip_features"
MODEL_DIR = PROJECT_ROOT / "models"
LABELS = sorted(
    {
        path.name.split("_")[0]
        for path in FEATURE_DIR.glob("*.npy")
    }
)
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}
EPOCHS = 200
CALIBRATION_REGISTRY = PROJECT_ROOT / "features" / "court" / "calibrations.json"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ShotDataset(Dataset):
    def __init__(self, feature_dir, calibration_registry=CALIBRATION_REGISTRY):
        self.feature_paths = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(".npy")]
        self.labels = [LABEL_TO_IDX[os.path.basename(f).split("_")[0]] for f in self.feature_paths]
        registry = CalibrationRegistry(calibration_registry)
        self.source_groups = [
            registry.source_for_clip(os.path.basename(path))
            for path in self.feature_paths
        ]

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        x = torch.tensor(np.load(self.feature_paths[idx]), dtype=torch.float32)  # [T, D]
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded = pad_sequence(list(sequences), batch_first=True)  # [B, T, D]
    lengths = torch.tensor([seq.size(0) for seq in sequences])
    labels = torch.stack(labels)
    return padded, lengths, labels


def grouped_train_val_split(dataset, val_size=0.2, random_state=42):
    groups = np.asarray(dataset.source_groups)
    if len(set(groups)) < 2:
        raise ValueError("grouped validation requires at least two source videos")
    indices = np.arange(len(dataset))
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=val_size, random_state=random_state
    )
    train_indices, val_indices = next(
        splitter.split(indices, np.asarray(dataset.labels), groups)
    )
    if set(groups[train_indices]) & set(groups[val_indices]):
        raise RuntimeError("source-video leakage detected in grouped split")
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

class ShotClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout_prob=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # Apply LayerNorm to the LSTM output
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = self.layer_norm(out)  # Normalize LSTM output
        last = out[range(len(lengths)), lengths - 1]  # [B, H*2]
        last = self.dropout(last)  # Apply dropout
        return self.fc(last)

if __name__ == "__main__":
    if not LABELS:
        raise FileNotFoundError(
            f"no generated clip features found in {FEATURE_DIR}; "
            "run src/extract_clip_features.py first"
        )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info("Using device: %s", device)

    dataset = ShotDataset(FEATURE_DIR)
    train_ds, val_ds = grouped_train_val_split(dataset)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    example_input, _, _ = next(iter(train_loader))

    input_dim = example_input.shape[2]
    model = ShotClassifier(
        input_dim=input_dim,
        hidden_dim=128,
        num_classes=len(LABELS),
        dropout_prob=0.5,
    ).to(device)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Weighted loss function to handle class imbalance
    class_counts = np.zeros(len(LABELS), dtype=np.int32)
    for _, label in dataset:
        class_counts[label.item()] += 1

    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # Weighted loss function, label smoothing to reduce model confidence
    # WOAH label smoothing did a lot of improvement

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    val_accuracies = []

    # Early stopping configuration
    patience = 20  # Number of epochs to wait for improvement
    no_improvement_epochs = 0

    best_f1_score = 0  # Track the best F1-score

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        total_loss = 0
        for x, lengths, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths.cpu())
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping to prevent exploding gradients
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Save model checkpoint
        checkpoint_dir = MODEL_DIR / "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"shot_classifier_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Model checkpoint saved to {checkpoint_path}")

        # Evaluation phase
        model.eval()
        correct = 0
        total = 0
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, lengths, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x, lengths.cpu())
                loss = criterion(logits, y)
                val_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        accuracy = correct / total
        val_accuracies.append(accuracy)

        # Calculate F1-score
        f1 = f1_score(all_labels, all_preds, average="weighted")
        logging.info(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {accuracy:.2%}, F1-Score = {f1:.4f}")

        # Early stopping logic based on F1-score
        if f1 > best_f1_score:
            best_f1_score = f1
            no_improvement_epochs = 0  # Reset counter
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model_path = MODEL_DIR / "shot_classifier.pth"
            torch.save(model.state_dict(), model_path)
            logging.info(f"Best model saved to {model_path} with F1-Score = {f1:.4f}")
        else:
            no_improvement_epochs += 1
            logging.info(f"No improvement in F1-Score for {no_improvement_epochs} epochs.")

        if no_improvement_epochs >= patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs due to no improvement in F1-Score.")
            break

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

    # Plot metrics
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_losses) + 1)

    # Plot training loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    # Plot validation loss
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()

    # Plot validation accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", marker="o", color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(MODEL_DIR / "training_metrics.png")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(LABELS)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.savefig(MODEL_DIR / "confusion_matrix.png")
    plt.show()
