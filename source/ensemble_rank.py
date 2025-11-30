#%% SETUP
import pandas as pd
import numpy as np
import os
import glob
import re
import xgboost as xgb
from catboost import CatBoostRegressor
import pickle

# --- CONFIGURATION ---
OUTPUT_DIR = os.path.join("..", "output")
SUBMISSION_DIR = os.path.join("..", "submissions")
DATA_DIR = os.path.join("..", "prc-2025-datasets-local")
TEAM_NAME = "merry-jacket" 

# --- INPUTS ---
RANK_FILE = os.path.join(OUTPUT_DIR, "cat_features_RANK.parquet")
TEMPLATE_RANK = os.path.join(DATA_DIR, "fuel_rank_submission.parquet")

MODEL_CATBOOST_PATH = os.path.join(OUTPUT_DIR, "catboost_model_log_flow.cbm")
MODEL_XGBOOST_PATH = os.path.join(OUTPUT_DIR, "xgb_model_linear.json")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "xgb_linear_ohe.pkl")

# --- WEIGHTS ---
WEIGHTS = [0.70, 0.30]

os.makedirs(SUBMISSION_DIR, exist_ok=True)

def get_next_filename(team, directory):
    pattern = os.path.join(directory, f"{team}_v*.parquet")
    existing = glob.glob(pattern)
    max_v = 0
    for f in existing:
        match = re.search(r"_v(\d+)\.parquet$", os.path.basename(f))
        if match:
            v = int(match.group(1))
            max_v = max(max_v, v)
    return f"{team}_v{max_v + 1}.parquet"

OUTPUT_FILENAME = get_next_filename(TEAM_NAME, SUBMISSION_DIR)
print(f"[INIT] Ensemble Ranking Script. Target: {OUTPUT_FILENAME}")

#%% LOAD MODELS
print("[LOAD] Loading Models...")
if os.path.exists(MODEL_CATBOOST_PATH):
    model_cb = CatBoostRegressor()
    model_cb.load_model(MODEL_CATBOOST_PATH)
else:
    # Fallback if file moved
    fallback = os.path.join(OUTPUT_DIR, "catboost_model_log_flow.cbm")
    model_cb = CatBoostRegressor()
    model_cb.load_model(fallback)

model_xgb = xgb.XGBRegressor()
model_xgb.load_model(MODEL_XGBOOST_PATH)

with open(ENCODER_PATH, 'rb') as f:
    encoder = pickle.load(f)

# Feature definitions for XGBoost
CAT_COLS = ['aircraft_type', 'flight_phase', 'gear_config', 'icao_wtc', 'engine_class']
EXCLUDE = ['idx', 'flight_id', 'fuel_kg']

#%% PREDICTION FUNCTION
def predict_ensemble(feature_file, dataset_name):
    print(f"\n--- Processing {dataset_name} ---")
    df = pd.read_parquet(feature_file)
    
    # --- CATBOOST ---
    print("[A] Predicting CatBoost (Log-Flow)...")
    X_cb = df[model_cb.feature_names_]
    pred_log_rate = model_cb.predict(X_cb)
    pred_rate = np.expm1(pred_log_rate)
    duration_safe = df['duration'].clip(lower=1.0)
    pred_A = np.maximum(pred_rate * duration_safe, 0)
    
    # --- XGBOOST ---
    print("[B] Predicting XGBoost (Linear)...")
    num_cols = [c for c in df.columns if c not in CAT_COLS + EXCLUDE]
    df_xgb = df.copy()
    df_xgb[num_cols] = df_xgb[num_cols].fillna(0.0)
    
    encoded_cats = encoder.transform(df_xgb[CAT_COLS])
    encoded_feature_names = encoder.get_feature_names_out(CAT_COLS)
    X_num = df_xgb[num_cols].reset_index(drop=True)
    X_cat = pd.DataFrame(encoded_cats, columns=encoded_feature_names)
    X_xgb_full = pd.concat([X_num, X_cat], axis=1)
    
    xgb_features = model_xgb.get_booster().feature_names
    X_xgb = X_xgb_full.reindex(columns=xgb_features, fill_value=0)
    pred_B = np.maximum(model_xgb.predict(X_xgb), 0)
    
    # --- BLEND ---
    print(f"[BLEND] Averaging A ({WEIGHTS[0]}) + B ({WEIGHTS[1]})...")
    pred_final = (pred_A * WEIGHTS[0]) + (pred_B * WEIGHTS[1])
    
    print(f"   Mean Prediction: {pred_final.mean():.2f} kg")
    
    return pd.DataFrame({'idx': df['idx'], 'fuel_kg_pred': pred_final})

#%% EXECUTE
df_res_rank = predict_ensemble(RANK_FILE, "RANKING DATA")

print("[MERGE] Aligning with Ranking Template...")
df_template = pd.read_parquet(TEMPLATE_RANK)
df_submission = pd.merge(df_template, df_res_rank, on='idx', how='left')

missing = df_submission['fuel_kg_pred'].isna().sum()
if missing > 0:
    print(f"[WARN] Filling {missing} missing rows.")
    df_submission['fuel_kg_pred'].fillna(df_submission['fuel_kg_pred'].mean(), inplace=True)

df_submission['fuel_kg'] = df_submission['fuel_kg_pred']
save_path = os.path.join(SUBMISSION_DIR, OUTPUT_FILENAME)

cols_out = ['idx', 'flight_id', 'start', 'end', 'fuel_kg']
df_submission[cols_out].to_parquet(save_path, index=False)

print(f"\n[DONE] Saved to: {save_path}")
