#%% SETUP
import pandas as pd
import numpy as np
import os
import xgboost as xgb
import optuna
import pickle
import time
from multiprocessing import cpu_count
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from functools import partial

# --- CONFIGURATION ---
OUTPUT_DIR = os.path.join("..", "output")
# Input: The Clean CatBoost features (Strings + Numerics)
TRAIN_FILE = os.path.join(OUTPUT_DIR, "cat_features_TRAIN.parquet")
# Output: XGBoost Linear Model, we process data only once for CB and adapt to XG
MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "xgb_model_linear.json")
ENCODER_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "xgb_linear_ohe.pkl")

N_TRIALS = 30           
CV_FOLDS = 3            
RANDOM_STATE = 42
N_PARALLEL_JOBS = max(1, cpu_count() - 2) 
TIMEOUT_MINUTES = 60 

optuna.logging.set_verbosity(optuna.logging.WARNING)

print(f"[INIT] Parallel Jobs: {N_PARALLEL_JOBS}, Timeout: {TIMEOUT_MINUTES} mins")

#%% DATA LOADING & PREPROCESSING
print(f"[LOAD] Reading training data...")
df_train = pd.read_parquet(TRAIN_FILE)

# 1. Define Feature Types
CAT_COLS = ['aircraft_type', 'flight_phase', 'gear_config', 'icao_wtc', 'engine_class']
EXCLUDE = ['idx', 'flight_id', 'fuel_kg']
# Identify numeric columns dynamically
NUM_COLS = [c for c in df_train.columns if c not in CAT_COLS + EXCLUDE]

# 2. ZERO IMPUTATION (For missing kinematics)
print("[PREP] Enforcing Zero Imputation on Numerics...")
df_train[NUM_COLS] = df_train[NUM_COLS].fillna(0.0)

# 3. ONE-HOT ENCODING
print("[PREP] Fitting OneHotEncoder on Categoricals...")
# handle_unknown='ignore' prevents crashes if future data has new categories
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.int8)
encoded_cats = encoder.fit_transform(df_train[CAT_COLS])

# Save the encoder for the submission script
with open(ENCODER_OUTPUT_PATH, 'wb') as f:
    pickle.dump(encoder, f)
print(f"[SAVE] Encoder saved to {ENCODER_OUTPUT_PATH}")

# 4. Assemble X and Y
encoded_feature_names = encoder.get_feature_names_out(CAT_COLS)
X_num = df_train[NUM_COLS].reset_index(drop=True)
X_cat = pd.DataFrame(encoded_cats, columns=encoded_feature_names)
X = pd.concat([X_num, X_cat], axis=1)

# Target is RAW FUEL (Linear Strategy)
Y = df_train['fuel_kg']
GROUPS = df_train['flight_id'] # Critical for GroupKFold

print(f"[DATA] Training Matrix: {X.shape}")
print(f"[DATA] Target Mean (Linear): {Y.mean():.2f} kg")

#%% OPTUNA OBJECTIVE 
def objective(trial, X, Y, groups, cv_folds, random_state):
    
    params = {
        # --- STRUCTURAL ---
        'max_depth': trial.suggest_int('max_depth', 6, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.1, 5.0),
        
        # --- SCALING ---
        'n_estimators': trial.suggest_int('n_estimators', 500, 2500), 
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        
        # --- REGULARIZATION ---
        'subsample': 0.8,           
        'colsample_bytree': 0.8,    
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        
        'random_state': random_state,
        'objective': 'reg:squarederror', 
        'tree_method': 'hist', 
        # IMPORTANT: n_jobs=1 inside the trial because we run trials in parallel
        'n_jobs': 1, 
        # Early stopping handled in fit()
    }
    
    # Use GroupKFold for true independence
    gkf = GroupKFold(n_splits=cv_folds)
    scores = []
    
    # Run 1 fold for speed during tuning (Validation on unseen flights)
    for train_idx, val_idx in gkf.split(X, Y, groups=groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = Y.iloc[train_idx], Y.iloc[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        preds = model.predict(X_val)
        preds = np.maximum(preds, 0) # Physics check
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)
        break # Stop after 1 fold to save time
        
    return np.mean(scores)

def print_progress(study, trial):
    print(f"[PROGRESS] Trial {len(study.trials)}/{N_TRIALS}. Best Linear RMSE: {study.best_value:.2f} kg")

#%% EXECUTE TUNING
print(f"\n[TUNING] Starting XGBoost Linear Search ({N_TRIALS} trials)...")
start_time = time.time()

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))

objective_with_data = partial(
    objective, X=X, Y=Y, groups=GROUPS, 
    cv_folds=CV_FOLDS, random_state=RANDOM_STATE
)

# PARALLEL EXECUTION
study.optimize(
    objective_with_data, 
    n_trials=N_TRIALS, 
    n_jobs=N_PARALLEL_JOBS, 
    callbacks=[print_progress],
    timeout=TIMEOUT_MINUTES * 60 
)

print(f"\n[TUNING] Done in {(time.time() - start_time)/60:.2f} minutes.")
print(f"[RESULT] Best Linear RMSE: {study.best_value:.2f} kg")

#%% FINAL TRAINING
best_params = study.best_params
# Boost iterations for final model on full data
final_iterations = int(best_params.get('n_estimators', 1000) * 1.2) 

final_params = {
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'n_jobs': -1 # Use ALL cores for final single training
}
final_params.update(best_params)
final_params['n_estimators'] = final_iterations

print(f"\n[TRAINING] Training Final XGBoost Linear Model ({final_iterations} trees)...")
final_model = xgb.XGBRegressor(**final_params)
final_model.fit(X, Y)

final_model.save_model(MODEL_OUTPUT_PATH)
print(f"[SAVE] Model saved to {MODEL_OUTPUT_PATH}")

# Feature Importance
imp = pd.Series(final_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Feature Importance (Top 10) ---")
print(imp.head(10))

#%% DIAGNOSTIC
print("\n" + "="*50 + "\nDIAGNOSTIC ERROR ANALYSIS\n" + "="*50)

Y_pred = np.maximum(final_model.predict(X), 0)
rmse_real = np.sqrt(mean_squared_error(Y, Y_pred))
print(f"[RESULT] Final Training RMSE: {rmse_real:.2f} kg")

df_analysis = df_train[['idx', 'fuel_kg', 'is_missing_data', 'MTOW_kg', 'aircraft_type']].copy()
df_analysis['pred'] = Y_pred
df_analysis['error'] = df_analysis['pred'] - df_analysis['fuel_kg']
df_analysis['abs_error'] = df_analysis['error'].abs()

print("\n--- Top 5 Worst Failures ---")

print(df_analysis.sort_values('abs_error', ascending=False).head(5))
