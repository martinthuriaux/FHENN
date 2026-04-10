import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# choose the dataset
dataset = pd.read_csv("sp100_dataset.csv")

# make sure date is datetime
dataset["date"] = pd.to_datetime(dataset["date"])

# define features and target
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
    "high_low_range_1d",
    "market_ret_5d",
    "volume_z_20d",
    "range_5d_avg",
    "ret_1d_rank",
]
target_col = "target_up_5d"

# split dataset into train/val/test based on date
train_df = dataset[dataset["date"] < "2023-01-01"].copy()
val_df = dataset[(dataset["date"] >= "2023-01-01") & (dataset["date"] < "2024-01-01")].copy()
test_df = dataset[dataset["date"] >= "2024-01-01"].copy()

# prepare data for modeling
X_train = train_df[feature_cols].values
X_val = val_df[feature_cols].values
X_test = test_df[feature_cols].values

y_train = train_df[target_col].values
y_val = val_df[target_col].values
y_test = test_df[target_col].values

# standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# train logistic regression
clf = LogisticRegression(max_iter=10000)
clf.fit(X_train_scaled, y_train)

# evaluate on validation and test sets
val_pred = clf.predict(X_val_scaled)
val_prob = clf.predict_proba(X_val_scaled)[:, 1]

test_pred = clf.predict(X_test_scaled)
test_prob = clf.predict_proba(X_test_scaled)[:, 1]

print("Validation Accuracy:", accuracy_score(y_val, val_pred))
print("Validation AUC:", roc_auc_score(y_val, val_prob))
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print("Test AUC:", roc_auc_score(y_test, test_prob))




