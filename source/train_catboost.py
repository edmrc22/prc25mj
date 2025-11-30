#%% SETUP
import pandas as pd
import numpy as np
import os
import time
import shutil
from datetime import datetime
import optuna
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from functools import partial

# --- CONFIGURATION ---
OUTPUT_DIR = os.path.join("..", "output")
TRAIN_FILE = os.path.join(OUTPUT_DIR, "cat_features_TRAIN.parquet")
MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "catboost_model_log_flow.cbm")

# --- TUNING SETTINGS ---
N_TRIALS = 30           
CV_FOLDS = 3            
RANDOM_STATE = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)

TASK_TYPE = 'GPU'
DEVICES = '0'

#%% DATA LOADING
print(f"[LOAD] Reading training data...")
df_train = pd.read_parquet(TRAIN_FILE)

CAT_COL_NAMES = ['aircraft_type', 'flight_phase', 'gear_config', 'icao_wtc', 'engine_class']
EXCLUDE_COLS = ['idx', 'flight_id', 'fuel_kg']
feature_cols = [c for c in df_train.columns if c not in EXCLUDE_COLS]

X = df_train[feature_cols]
GROUPS = df_train['flight_id'] 

# --- TARGET: LOG FLOW RATE ---
print("[PREP] Calculating Log-Flow Target...")
MIN_DURATION = 1.0
duration_safe = df_train['duration'].clip(lower=MIN_DURATION)
Y_flow_rate = df_train['fuel_kg'] / duration_safe
Y_log_flow = np.log1p(Y_flow_rate)

# CatBoost Indices
cat_indices = [X.columns.get_loc(c) for c in CAT_COL_NAMES if c in X.columns]

print(f"[DATA] Matrix: {X.shape}")
print(f"[DATA] Target Mean: {Y_log_flow.mean():.4f}")

#%% OPTUNA OBJECTIVE 
def objective(trial, X, Y, groups, cat_idx, cv_folds, random_state):
    
    params = {
        'boosting_type': 'Plain',       
        'bootstrap_type': 'Bernoulli',  
        'subsample': 0.8,               
        'border_count': 128,            
        
        # --- TUNABLE PARAMETERS ---
        'depth': trial.suggest_int('depth', 6, 10), 
        'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 15), 
        'random_strength': trial.suggest_float('random_strength', 1e-9, 10, log=True),
        
        # --- FIXED ---
        'iterations': 2000,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': random_state,
        'cat_features': cat_idx,
        'task_type': TASK_TYPE,
        'devices': DEVICES,
        'verbose': False
    }
    
    # GroupKFold
    gkf = GroupKFold(n_splits=cv_folds)
    scores = []
    
    for train_idx, val_idx in gkf.split(X, Y, groups=groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = Y.iloc[train_idx], Y.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        
        try:
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
            preds = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            scores.append(rmse)
        except Exception as e:
            print(f"Trial Err: {e}")
            return 1000.0
        
        break 
        
    return np.mean(scores)

def print_progress(study, trial):
    print(f"[PROGRESS] Trial {len(study.trials)}/{N_TRIALS}. Best Val RMSE: {study.best_value:.4f}")

#%% EXECUTE TUNING
print(f"\n[TUNING] Starting Optimized CatBoost Search (N={N_TRIALS})...")
start_time = time.time()

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))

objective_with_data = partial(
    objective, 
    X=X, Y=Y_log_flow, groups=GROUPS, cat_idx=cat_indices, 
    cv_folds=CV_FOLDS, random_state=RANDOM_STATE
)

study.optimize(objective_with_data, n_trials=N_TRIALS, n_jobs=1, callbacks=[print_progress])

print(f"\n[TUNING] Done in {(time.time() - start_time)/60:.2f} minutes.")
print(f"[RESULT] Best Params: {study.best_params}")

#%% FINAL MODEL TRAINING
best_params = study.best_params

# Boost iterations for final training
final_iterations = max(3000, int(best_params.get('iterations', 2000) * 1.5))

final_params = {
    'iterations': final_iterations,
    'boosting_type': 'Plain',
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
    'border_count': 128,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': RANDOM_STATE,
    'cat_features': cat_indices,
    'verbose': 500,
    'task_type': TASK_TYPE,
    'devices': DEVICES
}
# Merge tuned params
for k, v in best_params.items():
    if k != 'iterations': final_params[k] = v

print(f"\n[TRAINING] Training Final Model on Full Data ({final_iterations} iters)...")
final_model = CatBoostRegressor(**final_params)
final_model.fit(X, Y_log_flow)

final_model.save_model(MODEL_OUTPUT_PATH)
print(f"[SAVE] Model saved to {MODEL_OUTPUT_PATH}")

#%% CELL 6: DIAGNOSTIC (Real Scale)
print("\n" + "="*50 + "\nDIAGNOSTIC ERROR ANALYSIS (Linear Scale)\n" + "="*50)

pred_log_rate = final_model.predict(X)
pred_rate = np.expm1(pred_log_rate)
Y_pred_real = pred_rate * df_train['duration'] 
Y_pred_real = np.maximum(Y_pred_real, 0)

rmse_real = np.sqrt(mean_squared_error(df_train['fuel_kg'], Y_pred_real))
print(f"[RESULT] Final Training RMSE: {rmse_real:.2f} kg")

# Breakdown
df_analysis = X.copy()
df_analysis['actual'] = df_train['fuel_kg']
df_analysis['pred'] = Y_pred_real
df_analysis['error'] = df_analysis['pred'] - df_analysis['actual']
df_analysis['squared_error'] = df_analysis['error'] ** 2

print("\n--- RMSE Breakdown ---")
for status in [0, 1]:
    subset = df_analysis[df_analysis['is_missing_data'] == status]
    if len(subset) > 0:
        rmse = np.sqrt(subset['squared_error'].mean())
        label = "REAL TRAJECTORY" if status == 0 else "MISSING/IMPUTED"

        print(f"{label:<20} | Count: {len(subset):>6} | RMSE: {rmse:.2f} kg")
