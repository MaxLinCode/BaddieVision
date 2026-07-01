import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

# Load dataset
INPLAY_DIR = Path(__file__).resolve().parent
data = np.load(INPLAY_DIR / "combined_features.npy")
X = data[:, :-1].astype(np.float32)
y = data[:, -1].astype(np.float32)

from collections import Counter
print("Label distribution:", Counter(y.astype(int)))

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train).unsqueeze(1)
X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val).unsqueeze(1)

# Define upgraded model with dropout and deeper layers
input_dim = X.shape[1]
model = nn.Sequential(
    nn.Linear(input_dim, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)  # No sigmoid here
)

# Use BCEWithLogitsLoss (includes sigmoid internally)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=20)

# Training loop with early stopping
epochs = 1000
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 200
no_improve_counter = 0

for epoch in range(epochs):
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    train_losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = loss_fn(val_pred, y_val)
        val_losses.append(val_loss.item())
        acc = ((torch.sigmoid(val_pred) > 0.5) == y_val).float().mean().item()
        print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f} | Val Acc: {acc:.3f}")

    scheduler.step(val_loss)

    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        no_improve_counter = 0
    else:
        no_improve_counter += 1
        if no_improve_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
print("best_val_loss", best_val_loss)
# Plot loss curve
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# Evaluate on validation set with threshold tuning (raw only)
model.eval()
with torch.no_grad():
    logits = model(X_val).squeeze()
    probs = torch.sigmoid(logits).numpy()

best_f1 = 0
best_thresh = 0.5
for t in np.arange(0.3, 0.7, 0.01):
    raw_preds = (probs > t).astype(int)
    f1 = f1_score(y_val.numpy().astype(int), raw_preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

final_preds = (probs > best_thresh).astype(int)
y_true = y_val.squeeze().numpy().astype(int)  # shape (N,)
final_acc = (final_preds == y_true).mean()
cm = confusion_matrix(y_val.numpy().astype(int), final_preds)

print(f"\nBest threshold: {best_thresh:.2f} | F1 score (raw): {best_f1:.4f} | Accuracy (raw): {final_acc:.4f}")
print("Confusion Matrix:")
print(cm)
