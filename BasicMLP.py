import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------
# 1) Load dataset
# ---------------------------
dataset = pd.read_csv("sp100_dataset.csv")
dataset["date"] = pd.to_datetime(dataset["date"])

feature_cols = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_20d",
    "vol_ratio_20d",
    "ma_gap_20d",
    "high_low_range_1d",
    "ret_10d",
    "ret_60d",
    "vol_5d",
    "vol_60d",
    "ma_gap_5d",
    "ma_gap_60d",
    "volume_z_20d",
    "range_5d_avg",
]

target_col = "target_up_5d"

train_df = dataset[dataset["date"] < "2023-01-01"].copy()
val_df = dataset[(dataset["date"] >= "2023-01-01") & (dataset["date"] < "2024-01-01")].copy()
test_df = dataset[dataset["date"] >= "2024-01-01"].copy()

X_train = train_df[feature_cols].values
X_val = val_df[feature_cols].values
X_test = test_df[feature_cols].values

y_train = train_df[target_col].values.astype(np.float32)
y_val = val_df[target_col].values.astype(np.float32)
y_test = test_df[target_col].values.astype(np.float32)


# ---------------------------
# 2) Scale inputs using train only
# ---------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ---------------------------
# 3) Convert to torch tensors
# ---------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# ---------------------------
# 4) Define MLP
# ---------------------------
class StockMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


model = StockMLP(input_dim=len(feature_cols))

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ---------------------------
# 5) Train
# ---------------------------
epochs = 30

for epoch in range(epochs):
    model.train()

    logits = model(X_train_t)
    loss = criterion(logits, y_train_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_loss = criterion(val_logits, y_val_t)

    if epoch % 5 == 0 or epoch == epochs - 1:
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Val Loss: {val_loss.item():.4f}"
        )


# ---------------------------
# 6) Evaluate
# ---------------------------
model.eval()
with torch.no_grad():
    val_logits = model(X_val_t).squeeze().numpy()
    test_logits = model(X_test_t).squeeze().numpy()

val_probs = 1 / (1 + np.exp(-val_logits))
test_probs = 1 / (1 + np.exp(-test_logits))

val_preds = (val_probs >= 0.5).astype(int)
test_preds = (test_probs >= 0.5).astype(int)

print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation AUC:", roc_auc_score(y_val, val_probs))
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print("Test AUC:", roc_auc_score(y_test, test_probs))