import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)


# ---------------------------
# Config
# ---------------------------
DATA_PATH = "sp100_dataset.csv"
TARGET_COL = "target_fwd_5d"

BATCH_SIZE = 512
MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# 1) Load dataset
# ---------------------------
dataset = pd.read_csv(DATA_PATH)
dataset["date"] = pd.to_datetime(dataset["date"])

feature_cols = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "vol_5d",
    "vol_20d",
    "vol_60d",
    "vol_ratio_20d",
    "ma_gap_5d",
    "ma_gap_10d",
    "ma_gap_20d",
    "ma_gap_60d",
    "high_low_range",
    "close_open_ratio",
    "high_open_ratio",
    "low_open_ratio",
    "close_high_ratio",
    "close_low_ratio",
    "market_ret_1d",
    "market_ret_5d",
    "market_ret_20d",
    "market_vol_20d",
    "market_ma_gap_20d",
    "market_breakout_20d",
    "breakout_20d",
    "ret_1d_vs_sector",
    "ret_5d_vs_sector",
    "ret_20d_vs_sector",
    "vol_20d_vs_sector",
    "breakout_20d_vs_sector",
    "volume_z_20d_vs_sector",
]

# keep only rows with full data
dataset = dataset.dropna(subset=feature_cols + [TARGET_COL]).copy()

# Time split
train_df = dataset[dataset["date"] < "2023-01-01"].copy()
val_df = dataset[(dataset["date"] >= "2023-01-01") & (dataset["date"] < "2024-01-01")].copy()
test_df = dataset[dataset["date"] >= "2024-01-01"].copy()

X_train = train_df[feature_cols].values.astype(np.float32)
X_val = val_df[feature_cols].values.astype(np.float32)
X_test = test_df[feature_cols].values.astype(np.float32)

y_train = train_df[TARGET_COL].values.astype(np.float32)
y_val = val_df[TARGET_COL].values.astype(np.float32)
y_test = test_df[TARGET_COL].values.astype(np.float32)

print(f"Train shape: {X_train.shape}")
print(f"Val shape:   {X_val.shape}")
print(f"Test shape:  {X_test.shape}")


# ---------------------------
# 2) Torch tensors / loaders
# ---------------------------
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)
test_ds = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)


# ---------------------------
# 3) Smaller MLP
# ---------------------------
class StockMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 16),
            nn.ReLU(),

            nn.Linear(16, 1),

        )

    def forward(self, x):
        return self.net(x)


model = StockMLP(input_dim=len(feature_cols), dropout=0.1).to(DEVICE)


# ---------------------------
# 4) Loss / optimizer
# ---------------------------

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ---------------------------
# 5) Helper functions
# ---------------------------
def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            preds = model(xb)              # shape: [batch, 1]
            loss = criterion(preds, yb)

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * xb.size(0)
        all_preds.append(preds.detach().cpu())
        all_targets.append(yb.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)

    all_preds = torch.cat(all_preds).squeeze().numpy()
    all_targets = torch.cat(all_targets).squeeze().numpy()

    # correlation is a better metric than accuracy for regression
    if len(all_preds) > 1 and np.std(all_preds) > 0 and np.std(all_targets) > 0:
        corr = np.corrcoef(all_preds, all_targets)[0, 1]
    else:
        corr = np.nan

    return avg_loss, corr, all_preds, all_targets

def top_k_return(df, preds, target_col, k=0.2):
    """
    Select the top k% of stocks per date using predicted scores,
    then compute their average ACTUAL forward return.

    df: dataframe containing at least ['date', target_col]
    preds: numpy array of model predictions, same row order as df
    target_col: actual forward return column, e.g. 'target_fwd_5d'
    k: fraction to select, e.g. 0.2 = top 20%
    """
    eval_df = df[["date", target_col]].copy()
    eval_df["pred"] = preds

    daily_returns = []

    for _, group in eval_df.groupby("date"):
        group = group.sort_values("pred", ascending=False)
        top_n = max(1, int(len(group) * k))
        top_group = group.head(top_n)
        daily_returns.append(top_group[target_col].mean())

    return float(np.mean(daily_returns))


def bottom_k_return(df, preds, target_col, k=0.2):
    """
    Select the bottom k% of stocks per date using predicted scores,
    then compute their average ACTUAL forward return.
    """
    eval_df = df[["date", target_col]].copy()
    eval_df["pred"] = preds

    daily_returns = []

    for _, group in eval_df.groupby("date"):
        group = group.sort_values("pred", ascending=False)
        bottom_n = max(1, int(len(group) * k))
        bottom_group = group.tail(bottom_n)
        daily_returns.append(bottom_group[target_col].mean())

    return float(np.mean(daily_returns))


def top_bottom_spread(df, preds, target_col, k=0.2):
    """
    Difference between top-k return and bottom-k return.
    A larger positive spread is better.
    """
    top_ret = top_k_return(df, preds, target_col, k=k)
    bottom_ret = bottom_k_return(df, preds, target_col, k=k)
    return top_ret - bottom_ret

# ---------------------------
# 6) Train with early stopping
# ---------------------------
best_state = None
best_val_corr = -np.inf
best_epoch = -1
patience_counter = 0

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, train_corr, _, _ = run_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_corr, val_preds, val_targets = run_epoch(model, val_loader, criterion, optimizer=None)

    improved = val_corr > best_val_corr
    if improved:
        best_val_corr = val_corr
        best_epoch = epoch
        best_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | Train Corr: {train_corr:.4f} | "
            f"Val Loss: {val_loss:.6f} | Val Corr: {val_corr:.4f}"
        )

    if patience_counter >= PATIENCE:
        print(
            f"Early stopping at epoch {epoch}. "
            f"Best epoch was {best_epoch} with Val Corr {best_val_corr:.4f}"
        )
        break

if best_state is not None:
    model.load_state_dict(best_state)

# ---------------------------
# 7) Final evaluation
# ---------------------------
# ---------------------------
# 7) Final evaluation
# ---------------------------
val_loss, val_corr, val_preds, val_targets = run_epoch(model, val_loader, criterion, optimizer=None)
test_loss, test_corr, test_preds, test_targets = run_epoch(model, test_loader, criterion, optimizer=None)

print(f"Final Validation Loss: {val_loss:.6f}")
print(f"Final Validation Corr: {val_corr:.4f}")
print(f"Final Test Loss: {test_loss:.6f}")
print(f"Final Test Corr: {test_corr:.4f}")

# Top-k / Bottom-k portfolio-style metrics
val_top_10 = top_k_return(val_df, val_preds, TARGET_COL, k=0.10)
val_top_20 = top_k_return(val_df, val_preds, TARGET_COL, k=0.20)
val_bottom_10 = bottom_k_return(val_df, val_preds, TARGET_COL, k=0.10)
val_bottom_20 = bottom_k_return(val_df, val_preds, TARGET_COL, k=0.20)

test_top_10 = top_k_return(test_df, test_preds, TARGET_COL, k=0.10)
test_top_20 = top_k_return(test_df, test_preds, TARGET_COL, k=0.20)
test_bottom_10 = bottom_k_return(test_df, test_preds, TARGET_COL, k=0.10)
test_bottom_20 = bottom_k_return(test_df, test_preds, TARGET_COL, k=0.20)

print(f"Val Top 10% Avg Return: {val_top_10:.6f}")
print(f"Val Top 20% Avg Return: {val_top_20:.6f}")
print(f"Val Bottom 10% Avg Return: {val_bottom_10:.6f}")
print(f"Val Bottom 20% Avg Return: {val_bottom_20:.6f}")
print(f"Val Top-Bottom Spread 10%: {(val_top_10 - val_bottom_10):.6f}")
print(f"Val Top-Bottom Spread 20%: {(val_top_20 - val_bottom_20):.6f}")

print(f"Test Top 10% Avg Return: {test_top_10:.6f}")
print(f"Test Top 20% Avg Return: {test_top_20:.6f}")
print(f"Test Bottom 10% Avg Return: {test_bottom_10:.6f}")
print(f"Test Bottom 20% Avg Return: {test_bottom_20:.6f}")
print(f"Test Top-Bottom Spread 10%: {(test_top_10 - test_bottom_10):.6f}")
print(f"Test Top-Bottom Spread 20%: {(test_top_20 - test_bottom_20):.6f}")