import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# === Load Data ===
sequence_length = 36
INPLAY_DIR = Path(__file__).resolve().parent
X_seq = np.load(INPLAY_DIR / "data" / "img_3418_X.npy").astype(np.float32)
y_seq = np.load(INPLAY_DIR / "data" / "img_3418_Y.npy").astype(np.float32)

# === Label smoothing around rally edges ===
def edge_label_smoothing(labels, edge_width=2, smoothing=0.1):
    smoothed = labels.copy()
    for i in range(labels.shape[0]):
        for t in range(edge_width):
            if t < sequence_length:
                # Early edge
                if smoothed[i, t] != smoothed[i, t + 1]:
                    smoothed[i, t] = smoothed[i, t] * (1 - smoothing) + 0.5 * smoothing
                # Late edge
                if smoothed[i, -t-1] != smoothed[i, -t-2]:
                    smoothed[i, -t-1] = smoothed[i, -t-1] * (1 - smoothing) + 0.5 * smoothing
    return smoothed

y_seq = edge_label_smoothing(y_seq, edge_width=4, smoothing=0.2)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Convert to tensors (no unsqueeze!)
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size)

# === Model ===
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2,
                            bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, 1)  # ×2 for bidirectional

    def forward(self, x):
        out, _ = self.lstm(x)  # shape: (B, T, 2H)
        out = self.dropout(out)
        return self.fc(out).squeeze(-1)  # shape: (B, T)

# === Training ===
if __name__ == "__main__":
    input_dim = X_seq.shape[2]
    model = LSTMClassifier(input_size=input_dim, hidden_size=64, num_layers=2)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    epochs = 100
    patience = 10
    no_improve_counter = 0
    best_val_loss = float('inf')
    best_model_state = None

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # === Validation ===
        model.eval()
        val_loss_total = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                val_loss = loss_fn(pred, y_batch)
                val_loss_total += val_loss.item()
                all_preds.append(torch.sigmoid(pred))
                all_labels.append(y_batch)

        avg_val_loss = val_loss_total / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # Flatten predictions and labels for metrics
        y_pred_flat = torch.cat(all_preds).numpy().flatten()
        y_true_flat = torch.cat(all_labels).numpy().flatten()
        acc = ((y_pred_flat > 0.5) == y_true_flat).mean()

        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.3f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                break

    # === Save Models ===
    save_dir = INPLAY_DIR / "models"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_model_state, os.path.join(save_dir, "best_model.pt"))
    torch.save(model.state_dict(), os.path.join(save_dir, "last_model.pt"))

    # === Plot loss curve ===
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.show()

    # === Evaluate with best threshold ===
    probs = y_pred_flat
    best_f1 = 0
    best_thresh = 0.5
    for t in np.arange(0.3, 0.7, 0.01):
        preds = (probs > t).astype(int)
        f1 = f1_score(y_true_flat.astype(int), preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    final_preds = (probs > best_thresh).astype(int)
    final_acc = (final_preds == y_true_flat).mean()
    cm = confusion_matrix(y_true_flat.astype(int), final_preds)

    print(f"\n✅ Best Threshold: {best_thresh:.2f}")
    print(f"F1 Score: {best_f1:.4f} | Accuracy: {final_acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
