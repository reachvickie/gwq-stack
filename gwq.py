!pip install catboost

import numpy as np
import pandas as pd
import joblib
import warnings
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

FILEPATH = "/content/drive/MyDrive/27800394/Dataset/Combined Data/Combined_dataset.csv"

df = pd.read_csv(FILEPATH)

df.columns = [c.strip().replace('(', '').replace(')','').replace('/', '_')
              .replace(' ', '_').replace('-', '_') for c in df.columns]

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce')
TARGET = "CCME_Values"
df = df.dropna(subset=[TARGET])

num_features = [
    "Ammonia_mg_l", "Biochemical_Oxygen_Demand_mg_l", "Dissolved_Oxygen_mg_l",
    "Orthophosphate_mg_l", "pH_ph_units", "Temperature_cel",
    "Nitrogen_mg_l", "Nitrate_mg_l"
]
cat_features = ["Country", "Area", "Waterbody_Type"]

for col in cat_features:
    df[col] = df[col].astype(str)
    df[f"{col}_encoded"] = LabelEncoder().fit_transform(df[col])

encoded_cat_cols = [f"{c}_encoded" for c in cat_features]

train_df = df[df["Date"] < "2015-01-01"].copy()
test_df  = df[df["Date"] >= "2015-01-01"].copy()

def cascade_impute(df_input):
    df_copy = df_input.copy()
    df_copy = df_copy.sort_values("Date")
    df_copy[num_features] = df_copy.groupby("Area")[num_features].ffill().bfill()
    for col in num_features:
        df_copy[col] = df_copy[col].fillna(df_copy.groupby("Waterbody_Type")[col].transform("median"))
    df_copy[num_features] = df_copy[num_features].fillna(df_copy[num_features].median())
    return df_copy

train_df = cascade_impute(train_df)
test_df  = cascade_impute(test_df)

# Prepare Arrays
X_train = train_df[num_features + encoded_cat_cols].values
y_train = train_df[TARGET].values
groups_train = train_df["Area_encoded"].values

X_test  = test_df[num_features + encoded_cat_cols].values
y_test  = test_df[TARGET].values

def get_base_models_gpu():
    return (
        # XGBoost 
        xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05,
                         tree_method='hist', device='cuda'),

        # CatBoost 
        CatBoostRegressor(iterations=500, depth=8, learning_rate=0.05,
                          task_type="GPU", verbose=0),

        # LightGBM 
        lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                          device="gpu", verbosity=-1),

        # XGBRF 
        xgb.XGBRFRegressor(n_estimators=100, tree_method='hist', device='cuda')
    )

gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros((X_train.shape[0], 4))

print("Starting GPU-Accelerated GroupKFold Stacking...")
for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups=groups_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    m1, m2, m3, m4 = get_base_models_gpu()

    m1.fit(X_tr, y_tr)
    m2.fit(X_tr, y_tr)
    m3.fit(X_tr, y_tr)
    m4.fit(X_tr, y_tr)

    oof_preds[val_idx, 0] = m1.predict(X_val)
    oof_preds[val_idx, 1] = m2.predict(X_val)
    oof_preds[val_idx, 2] = m3.predict(X_val)
    oof_preds[val_idx, 3] = m4.predict(X_val)
    print(f"Fold {fold+1} complete.")

scaler = StandardScaler()
X_meta_train = scaler.fit_transform(oof_preds)

print("\nFitting final base models on full training set")
full_m1, full_m2, full_m3, full_m4 = get_base_models_gpu()
full_m1.fit(X_train, y_train)
full_m2.fit(X_train, y_train)
full_m3.fit(X_train, y_train)
full_m4.fit(X_train, y_train)

# meta-features for test set
test_base_preds = np.column_stack([
    full_m1.predict(X_test),
    full_m2.predict(X_test),
    full_m3.predict(X_test),
    full_m4.predict(X_test)
])
X_meta_test = scaler.transform(test_base_preds)

# architectures for empirical justification
meta_models = {
    "Linear Baseline": LinearRegression(),
    "Shallow MLP (16)": MLPRegressor(hidden_layer_sizes=(16,), max_iter=300, random_state=42),
    "Proposed Deep MLP (64, 32)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300, random_state=42)
}

results = []

print("\n--- Meta-Learner Performance Comparison ---")
for name, model in meta_models.items():
    model.fit(X_meta_train, y_train)
    y_pred = model.predict(X_meta_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({"Architecture": name, "RMSE": rmse, "MAE": mae, "R2": r2})

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

final_pipeline = {
    "base_models": {"xgb": full_m1, "cat": full_m2, "lgb": full_m3, "rf": full_m4},
    "meta_learner": meta_models["Proposed Deep MLP (64, 32)"],
    "scaler": scaler,
    "features": {"num": num_features, "cat": encoded_cat_cols}
}
joblib.dump(final_pipeline, "/content/drive/MyDrive/gwq.pkl")
print("\nPipeline saved to /content/drive/MyDrive/gwq.pkl")

