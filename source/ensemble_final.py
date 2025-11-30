#%% SETUP
import pandas as pd
import numpy as np
import os
import xgboost as xgb
from catboost import CatBoostRegressor
import pickle

# --- CONFIGURATION ---
OUTPUT_DIR = os.path.join("..", "output")
SUBMISSION_DIR = os.path.join("..", "submissions")
DATA_DIR = os.path.join("..", "prc-2025-datasets-local")
TEAM_NAME = "merry-jacket" 

# --- INPUTS ---
RANK_FEAT_FILE = os.path.join(OUTPUT_DIR, "cat_features_RANK.parquet")
FINAL_FEAT_FILE = os.path.join(OUTPUT_DIR, "cat_features_FINAL.parquet")
TEMPLATE_FINAL = os.path.join(DATA_DIR, "fuel_final_submission.parquet")

# --- MODELS ---
CATBOOST_MODEL_PATH = os.path.join(OUTPUT_DIR, "catboost_model_log_flow.cbm")
XGBOOST_MODEL_PATH = os.path.join(OUTPUT_DIR, "xgb_model_linear.json")
ENCODER_PATH = os.path.join(OUTPUT_DIR, "xgb_linear_ohe.pkl")

# --- WEIGHTS ---
WEIGHTS = [0.70, 0.30]

CAT_COLS = ['aircraft_type', 'flight_phase', 'gear_config', 'icao_wtc', 'engine_class']
EXCLUDE = ['idx', 'flight_id', 'fuel_kg']
REQUIRED_COLS = ['idx', 'flight_id', 'start', 'end', 'fuel_kg']
OUTPUT_FILENAME = "merry-jacket_final.parquet"

os.makedirs(SUBMISSION_DIR, exist_ok=True)
print(f"[INIT] Direct Prediction Script.")

#%% LOAD ASSETS
print("[LOAD] Models...")
model_cb = CatBoostRegressor()
model_cb.load_model(CATBOOST_MODEL_PATH)
model_xgb = xgb.XGBRegressor()
model_xgb.load_model(XGBOOST_MODEL_PATH)
with open(ENCODER_PATH, 'rb') as f: encoder = pickle.load(f)
print("[OK] Assets Ready.")

#%% PREDICTION ENGINE (Returns NumPy Array)
def get_prediction_array(file_path, label):
    """ Reads a feature file and returns a NumPy array of predictions (NO IDs). """
    if not os.path.exists(file_path):
        print(f"[FAIL] {label} missing: {file_path}")
        return np.array([])

    print(f"\n--- Processing {label} ---")
    df = pd.read_parquet(file_path)
    print(f"   Input Rows: {len(df)}")

    # --- CatBoost ---
    for c in CAT_COLS:
        if c in df.columns: df[c] = df[c].fillna("UNKNOWN").astype(str)
    X_cb = df[model_cb.feature_names_]
    pred_rate = np.expm1(model_cb.predict(X_cb))
    duration = df['duration'].clip(lower=1.0)
    pred_A = np.maximum(pred_rate * duration, 0)

    # --- XGBoost ---
    num_cols = [c for c in df.columns if c not in CAT_COLS + EXCLUDE]
    df_xgb = df.copy()
    df_xgb[num_cols] = df_xgb[num_cols].fillna(0.0)
    
    encoded_cats = encoder.transform(df_xgb[CAT_COLS])
    X_num = df_xgb[num_cols].reset_index(drop=True)
    X_cat = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(CAT_COLS))
    X_full = pd.concat([X_num, X_cat], axis=1)
    
    X_aligned = X_full.reindex(columns=model_xgb.get_booster().feature_names, fill_value=0)
    pred_B = np.maximum(model_xgb.predict(X_aligned), 0)

    # --- BLEND & RETURN ---
    return (pred_A * WEIGHTS[0]) + (pred_B * WEIGHTS[1])

#%% THE MERGE (ROW ORDER ASSIGNMENT)
# 1. Generate Raw Prediction Arrays
preds_rank = get_prediction_array(RANK_FEAT_FILE, "PHASE 1 (Rank)")
preds_final = get_prediction_array(FINAL_FEAT_FILE, "PHASE 2 (Final)")

# 2. Safety Check
if len(preds_rank) + len(preds_final) != 61745:
    print(f"[FATAL] Feature file row count mismatch! Expected 61745, got {len(preds_rank) + len(preds_final)}")
    exit()

# 3. Array Concatenation (The clean merge)
# This assumes the template order is [Phase 1 | Phase 2]
print("\n[STACK] Concatenating prediction arrays...")
all_preds_array = np.concatenate([preds_rank, preds_final])

# 4. Load Template
print(f"[MERGE] Loading Template: {os.path.basename(TEMPLATE_FINAL)}")
df_submission = pd.read_parquet(TEMPLATE_FINAL)
target_rows = len(df_submission)

# 5. ASSIGNMENT (THE FIX)
# We assign the prediction array directly to the template
df_submission['fuel_kg'] = all_preds_array

# 6. Final Clean & Save
if len(df_submission) != target_rows:
    print(f"[FATAL] Row count post-assignment is wrong. Aborting.")
    exit()

df_submission['fuel_kg'] = df_submission['fuel_kg'].fillna(df_submission['fuel_kg'].mean())
df_submission['fuel_kg'] = df_submission['fuel_kg'].clip(lower=1.0) # Clip near-zero predictions

output_path = os.path.join(SUBMISSION_DIR, OUTPUT_FILENAME)
df_submission[REQUIRED_COLS].to_parquet(output_path, index=False)

print(f"\n[DONE] File Generated.")
print(f"       File: {output_path}")

print(f"       Rows: {len(df_submission)}")
