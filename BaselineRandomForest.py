import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

dataset = pd.read_csv("sp100_dataset.csv")
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
    "high_low_range_1d",
    "market_ret_5d",
    "volume_z_20d",
    "range_5d_avg",
    "ret_1d_rank",
]

target_col = "target_up_5d"

train_df = dataset[dataset["date"] < "2023-01-01"].copy()
val_df = dataset[(dataset["date"] >= "2023-01-01") & (dataset["date"] < "2024-01-01")].copy()
test_df = dataset[dataset["date"] >= "2024-01-01"].copy()

X_train = train_df[feature_cols]
X_val = val_df[feature_cols]
X_test = test_df[feature_cols]

y_train = train_df[target_col]
y_val = val_df[target_col]
y_test = test_df[target_col]

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=50,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

val_pred = rf.predict(X_val)
val_prob = rf.predict_proba(X_val)[:, 1]

test_pred = rf.predict(X_test)
test_prob = rf.predict_proba(X_test)[:, 1]

print("Validation Accuracy:", accuracy_score(y_val, val_pred))
print("Validation AUC:", roc_auc_score(y_val, val_prob))
print("Test Accuracy:", accuracy_score(y_test, test_pred))
print("Test AUC:", roc_auc_score(y_test, test_prob))
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances)