import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------
# Reproducibility
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)


# ---------------------------
# Config
# ---------------------------
DATA_PATH = "sp100_dataset.csv"
TARGET_COL = "target_fwd_5d"

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

dataset = dataset.dropna(subset=feature_cols + [TARGET_COL]).copy()

# Time split
train_df = dataset[dataset["date"] < "2023-01-01"].copy()
val_df = dataset[(dataset["date"] >= "2023-01-01") & (dataset["date"] < "2024-01-01")].copy()
test_df = dataset[dataset["date"] >= "2024-01-01"].copy()

print(f"Train rows: {len(train_df)}")
print(f"Val rows:   {len(val_df)}")
print(f"Test rows:  {len(test_df)}")


# ---------------------------
# 2) Build daily groups
# ---------------------------
def make_daily_groups(df, feature_cols, target_col):
    groups = []
    for date, group in df.groupby("date"):
        x = torch.tensor(group[feature_cols].values, dtype=torch.float32)
        y = torch.tensor(group[target_col].values, dtype=torch.float32)
        groups.append((date, x, y, group.copy()))
    return groups

train_groups = make_daily_groups(train_df, feature_cols, TARGET_COL)
val_groups = make_daily_groups(val_df, feature_cols, TARGET_COL)
test_groups = make_daily_groups(test_df, feature_cols, TARGET_COL)

print(f"Train dates: {len(train_groups)}")
print(f"Val dates:   {len(val_groups)}")
print(f"Test dates:  {len(test_groups)}")


# ---------------------------
# 3) Model
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
        return self.net(x).squeeze(-1)  # shape [n]


model = StockMLP(input_dim=len(feature_cols), dropout=0.10).to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


# ---------------------------
# 4) Pairwise ranking loss
# ---------------------------
def pairwise_ranking_loss(scores: torch.Tensor, targets: torch.Tensor, min_gap: float = 0.0):
    """
    scores:  shape [n]
    targets: shape [n]
    min_gap: ignore pairs with tiny target differences
    """
    scores = scores.view(-1)
    targets = targets.view(-1)

    target_diff = targets[:, None] - targets[None, :]
    score_diff = scores[:, None] - scores[None, :]

    pair_mask = target_diff > min_gap

    if pair_mask.sum() == 0:
        return None

    loss = F.softplus(-score_diff[pair_mask]).mean()
    return loss


# ---------------------------
# 5) Helpers
# ---------------------------
def run_ranking_epoch(model, groups, optimizer=None, min_gap=0.0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    num_used = 0

    all_scores = []
    all_targets = []
    all_dates = []

    for date, xb, yb, group_df in groups:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            scores = model(xb)
            loss = pairwise_ranking_loss(scores, yb, min_gap=min_gap)

            if loss is None:
                continue

            if is_train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        num_used += 1

        all_scores.append(scores.detach().cpu().numpy())
        all_targets.append(yb.detach().cpu().numpy())
        all_dates.extend([date] * len(yb))

    avg_loss = total_loss / max(num_used, 1)

    if len(all_scores) > 0:
        all_scores = np.concatenate(all_scores)
        all_targets = np.concatenate(all_targets)
    else:
        all_scores = np.array([])
        all_targets = np.array([])

    if len(all_scores) > 1 and np.std(all_scores) > 0 and np.std(all_targets) > 0:
        corr = np.corrcoef(all_scores, all_targets)[0, 1]
    else:
        corr = np.nan

    return avg_loss, corr, all_scores, all_targets, np.array(all_dates)


def top_k_return_from_arrays(dates, scores, targets, k=0.2):
    eval_df = pd.DataFrame({
        "date": dates,
        "score": scores,
        "target": targets,
    })

    daily_returns = []

    for _, group in eval_df.groupby("date"):
        group = group.sort_values("score", ascending=False)
        top_n = max(1, int(len(group) * k))
        top_group = group.head(top_n)
        daily_returns.append(top_group["target"].mean())

    return float(np.mean(daily_returns))

def bottom_k_return_from_arrays(dates, scores, targets, k=0.2):
    eval_df = pd.DataFrame({
        "date": dates,
        "score": scores,
        "target": targets,
    })

    daily_returns = []

    for _, group in eval_df.groupby("date"):
        group = group.sort_values("score", ascending=False)
        bottom_n = max(1, int(len(group) * k))
        bottom_group = group.tail(bottom_n)
        daily_returns.append(bottom_group["target"].mean())

    return float(np.mean(daily_returns))


def top_bottom_spread_from_arrays(dates, scores, targets, k=0.2):
    top_ret = top_k_return_from_arrays(dates, scores, targets, k=k)
    bottom_ret = bottom_k_return_from_arrays(dates, scores, targets, k=k)
    return top_ret - bottom_ret

def sign_accuracy(scores, targets):
    pred_sign = (scores > 0).astype(int)
    true_sign = (targets > 0).astype(int)
    return accuracy_score(true_sign, pred_sign)


def sign_auc(scores, targets):
    true_sign = (targets > 0).astype(int)
    if len(np.unique(true_sign)) < 2:
        return np.nan
    return roc_auc_score(true_sign, scores)


def pairwise_accuracy(dates, scores, targets):
    eval_df = pd.DataFrame({
        "date": dates,
        "score": scores,
        "target": targets,
    })

    correct = 0
    total = 0

    for _, group in eval_df.groupby("date"):
        s = group["score"].values
        t = group["target"].values

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if t[i] == t[j]:
                    continue
                total += 1
                true_order = t[i] > t[j]
                pred_order = s[i] > s[j]
                if true_order == pred_order:
                    correct += 1

    return correct / total if total > 0 else np.nan


# ---------------------------
# 6) Train with early stopping
# ---------------------------
best_state = None
best_val_corr = -np.inf
best_epoch = -1
patience_counter = 0

for epoch in range(1, MAX_EPOCHS + 1):
    train_loss, train_corr, _, _, _ = run_ranking_epoch(
        model, train_groups, optimizer=optimizer, min_gap=0.0
    )
    val_loss, val_corr, val_scores, val_targets, val_dates = run_ranking_epoch(
        model, val_groups, optimizer=None, min_gap=0.0
    )

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
val_loss, val_corr, val_scores, val_targets, val_dates = run_ranking_epoch(
    model, val_groups, optimizer=None, min_gap=0.0
)
test_loss, test_corr, test_scores, test_targets, test_dates = run_ranking_epoch(
    model, test_groups, optimizer=None, min_gap=0.0
)

print(f"Final Validation Loss: {val_loss:.6f}")
print(f"Final Validation Corr: {val_corr:.4f}")
print(f"Final Test Loss: {test_loss:.6f}")
print(f"Final Test Corr: {test_corr:.4f}")

print("Val Sign Accuracy:", sign_accuracy(val_scores, val_targets))
print("Test Sign Accuracy:", sign_accuracy(test_scores, test_targets))

print("Val Sign AUC:", sign_auc(val_scores, val_targets))
print("Test Sign AUC:", sign_auc(test_scores, test_targets))

print("Val Pairwise Accuracy:", pairwise_accuracy(val_dates, val_scores, val_targets))
print("Test Pairwise Accuracy:", pairwise_accuracy(test_dates, test_scores, test_targets))

# Top / Bottom bucket returns
val_top_10 = top_k_return_from_arrays(val_dates, val_scores, val_targets, k=0.10)
val_top_20 = top_k_return_from_arrays(val_dates, val_scores, val_targets, k=0.20)
val_bottom_10 = bottom_k_return_from_arrays(val_dates, val_scores, val_targets, k=0.10)
val_bottom_20 = bottom_k_return_from_arrays(val_dates, val_scores, val_targets, k=0.20)

test_top_10 = top_k_return_from_arrays(test_dates, test_scores, test_targets, k=0.10)
test_top_20 = top_k_return_from_arrays(test_dates, test_scores, test_targets, k=0.20)
test_bottom_10 = bottom_k_return_from_arrays(test_dates, test_scores, test_targets, k=0.10)
test_bottom_20 = bottom_k_return_from_arrays(test_dates, test_scores, test_targets, k=0.20)

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