import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error

PIPELINE_PATH = "/gwq.pkl" #modify the pipeline path
pipeline = joblib.load(PIPELINE_PATH)

models = pipeline["base_models"]
meta_model = pipeline["meta_learner"]
scaler = pipeline["scaler"]

num_features = pipeline["features"]["num"]
cat_features = pipeline["features"]["cat"]

TARGET = "CCME_Values"

FILEPATH = "27800394/Dataset/Combined Data/Combined_dataset.csv" # set the proper dataset path

df = pd.read_csv(FILEPATH)

# Clean column names
df.columns = [
    c.strip()
     .replace('(', '')
     .replace(')', '')
     .replace('/', '_')
     .replace(' ', '_')
     .replace('-', '_')
    for c in df.columns
]

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

test_df = df[df["Date"] >= "2015-01-01"].copy()
test_df = test_df.sort_values("Date").reset_index(drop=True)


for col in ["Country", "Area", "Waterbody_Type"]:
    df[col] = df[col].astype("category")

    test_df[col] = pd.Categorical(
        test_df[col],
        categories=df[col].cat.categories
    )

    test_df[f"{col}_encoded"] = test_df[col].cat.codes

encoded_cat_cols = [f"{c}_encoded" for c in ["Country", "Area", "Waterbody_Type"]]


def predict_pipeline(X):
    p1 = models["xgb"].predict(X)
    p2 = models["cat"].predict(X)
    p3 = models["lgb"].predict(X)
    p4 = models["rf"].predict(X)

    meta_input = scaler.transform(np.column_stack([p1, p2, p3, p4]))
    return meta_model.predict(meta_input)



np.random.seed(42)

global_medians = test_df[num_features].median().to_dict()

def impute_none(df):
    d = df.copy()
    d[num_features] = d[num_features].fillna(0)
    return d

def impute_tier3(df):
    d = df.copy()
    d[num_features] = d[num_features].fillna(pd.Series(global_medians))
    return d

def impute_tier2(df):
    d = df.copy().sort_values("Date")
    d[num_features] = d.groupby("Area")[num_features].ffill()
    d[num_features] = d[num_features].fillna(pd.Series(global_medians))
    return d

def apply_dropout(df, cols, rate):
    d = df.copy()
    mask = np.random.rand(len(d)) < rate
    for col in cols:
        d.loc[mask, col] = np.nan
    return d, mask   # return mask for consistency


def evaluate(df):
    X = df[num_features + encoded_cat_cols]
    y = df[TARGET]
    y_pred = predict_pipeline(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae  = mean_absolute_error(y, y_pred)
    return rmse, mae

baseline_rmse, baseline_mae = evaluate(test_df)

print("\n=== BASELINE ===")
print(f"RMSE: {baseline_rmse:.5f} | MAE: {baseline_mae:.5f}")

scenarios = [
    (["Ammonia_mg_l"], "Ammonia (k=1)"),
    (["Orthophosphate_mg_l"], "Orthophosphate (k=1)"),
    (["Ammonia_mg_l", "Orthophosphate_mg_l"], "NH3 + PO4 (k=2)"),
    (["Ammonia_mg_l", "Biochemical_Oxygen_Demand_mg_l"], "NH3 + BOD (k=2)"),
    (["Ammonia_mg_l", "Orthophosphate_mg_l", "Biochemical_Oxygen_Demand_mg_l"],
     "NH3 + PO4 + BOD (k=3)")
]

drop_rates = [0.1, 0.3, 0.5]

results = []

print("\n=== FAULT TOLERANCE EXPERIMENT ===")

for cols, name in scenarios:
    print(f"\n--- {name} ---")

    for rate in drop_rates:

        # SAME dropout for all methods
        df_fault, mask = apply_dropout(test_df, cols, rate)

        # No imputation
        rmse_t0, _ = evaluate(impute_none(df_fault))

        # Tier-3
        rmse_t3, _ = evaluate(impute_tier3(df_fault))

        # Tier-2
        rmse_t2, _ = evaluate(impute_tier2(df_fault))

        print(f"\nDrop {int(rate*100)}%")
        print(f"  No-impute → {rmse_t0:.4f}")
        print(f"  Tier-3    → {rmse_t3:.4f}")
        print(f"  Tier-2    → {rmse_t2:.4f}")
        print(f"  Gain (T2 vs T3): {rmse_t3 - rmse_t2:.4f}")

        results.append({
            "Scenario": name,
            "Drop": rate,
            "Baseline": baseline_rmse,
            "No_Impute": rmse_t0,
            "Tier3": rmse_t3,
            "Tier2": rmse_t2,
            "Δ_T3": rmse_t3 - baseline_rmse,
            "Δ_T2": rmse_t2 - baseline_rmse,
            "Recovery": rmse_t3 - rmse_t2
        })

results_df = pd.DataFrame(results)

print("\n=== FINAL RESULTS TABLE ===")
print(results_df.to_string(index=False))

