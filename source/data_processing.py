#%% SETUP
import pandas as pd
import numpy as np
import os
import time
import math
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
DATA_DIR = os.path.join("..", "prc-2025-datasets-local")
OUTPUT_DIR = os.path.join("..", "output")
NUM_WORKERS = max(1, cpu_count() - 2)

FILE_TRAIN = os.path.join(OUTPUT_DIR, "cat_features_TRAIN.parquet")
FILE_RANK = os.path.join(OUTPUT_DIR, "cat_features_RANK.parquet")
FILE_FINAL = os.path.join(OUTPUT_DIR, "cat_features_FINAL.parquet")

LB_TO_KG = 0.453592
FT_TO_M = 0.3048
KNOT_TO_M_S = 0.514444
AIRCRAFT_SPEC_FILE = "aircraft_data.xlsx" 

print(f"[INIT] Workers: {NUM_WORKERS}")

# --- FEATURE INVENTORY ---
FINAL_COLS = [
    # A. Keys & Target
    'idx', 'flight_id', 'fuel_kg',
    
    # B. Categorical
    'aircraft_type', 'flight_phase', 'gear_config', 'icao_wtc', 'engine_class',
    
    # C. Static Context
    'route_distance_nm', 'elev_diff_ft', 'hour_sin', 'hour_cos',
    
    # D. Aircraft Specs (Numeric)
    'MTOW_kg', 'MALW_kg', 'wingspan_m', 'length_m', 'tail_height_m', 'wheelbase_m', 'approach_speed_ms',
    
    # E. Kinematics (0.0 if Blind)
    'duration', 'avg_altitude', 'avg_groundspeed', 'avg_TAS', 'avg_mach', 
    'avg_vert_rate', 'max_abs_vert_rate', 'delta_altitude',
    
    # F. Derived Proxies
    'time_into_flight_ratio', 'mass_ratio_proxy', 'aspect_ratio_proxy', 'tail_length_ratio',
    'weight_delta_ratio', 'energy_rate_index', 'mass_speed_index', 'distance_nm_approx',
    'is_missing_data'
]

# Categorical Columns for CatBoost Metadata (String)
CAT_FEATURES_LIST = ['aircraft_type', 'flight_phase', 'gear_config', 'icao_wtc', 'engine_class']

#%% HELPER FUNCTIONS
def haversine(lat1, lon1, lat2, lon2):
    R = 3440.065
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2): return np.nan
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    lambda1, lambda2 = math.radians(lon1), math.radians(lon2)
    a = math.sin((phi2-phi1)/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin((lambda2-lambda1)/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def calculate_isa_speed_of_sound(altitude_ft):
    H = altitude_ft * FT_TO_M
    T = 288.15 + (-0.0065 * H) if H <= 11000 else 216.65
    return math.sqrt(1.4 * 287.058 * T)

#%% WORKER FUNCTION 
def process_single_flight(flight_id, flightlist_entry, fuel_segments_df):
    features_list_local = []
    
    # 1. Static Context
    aircraft_type = str(flightlist_entry['aircraft_type'])
    route_distance_nm = flightlist_entry.get('route_distance_nm', np.nan)
    elev_diff_ft = flightlist_entry.get('elev_diff_ft', np.nan)
    takeoff_time = pd.to_datetime(flightlist_entry['takeoff'])
    total_flight_duration = (pd.to_datetime(flightlist_entry['landed']) - takeoff_time).total_seconds()
    
    # 2. Spec Lookups
    specs = {
        'MTOW_kg': flightlist_entry.get('MTOW_kg', np.nan),
        'MALW_kg': flightlist_entry.get('MALW_kg', np.nan),
        'approach_speed_ms': flightlist_entry.get('approach_speed_ms', np.nan),
        'tail_height_m': flightlist_entry.get('tail_height_m', np.nan),
        'wheelbase_m': flightlist_entry.get('wheelbase_m', np.nan),
        'wingspan_m': flightlist_entry.get('wingspan_m', np.nan),
        'length_m': flightlist_entry.get('length_m', np.nan),
        'gear_config': str(flightlist_entry.get('gear_config', 'UNKNOWN')),
        'icao_wtc': str(flightlist_entry.get('icao_wtc', 'UNKNOWN')),
        'engine_class': str(flightlist_entry.get('engine_class', 'UNKNOWN'))
    }

    # 3. Trajectory Loading
    traj_path = flightlist_entry['traj_path']
    traj_exists = os.path.exists(traj_path)
    df_traj = pd.read_parquet(traj_path) if traj_exists else None

    for _, row in fuel_segments_df.iterrows():
        t_start, t_end = pd.to_datetime(row['start']), pd.to_datetime(row['end'])
        row_idx = row['idx'] if 'idx' in row else np.nan
        fuel_kg = row['fuel_kg'] if 'fuel_kg' in row else np.nan
        duration = (t_end - t_start).total_seconds()
        
        # --- Data Availability ---
        has_data = False
        slice_data = None
        
        if traj_exists and df_traj is not None:
            mask = (df_traj['timestamp'] >= t_start) & (df_traj['timestamp'] <= t_end)
            slice_data = df_traj.loc[mask].copy()
            if not slice_data.empty:
                has_data = True
                slice_data.ffill(inplace=True)
                slice_data.fillna({'altitude': 0, 'groundspeed': 0, 'vertical_rate': 0, 'TAS': 0.0}, inplace=True)

        # --- Feature Calculation ---
        if has_data:
            # CASE A: Valid Data (Calculate normally)
            is_missing_data = 0
            avg_alt = slice_data['altitude'].mean()
            avg_speed = slice_data['groundspeed'].mean()
            avg_vert_rate = slice_data['vertical_rate'].mean()
            delta_alt = slice_data['altitude'].max() - slice_data['altitude'].min()
            max_abs_vert_rate = slice_data['vertical_rate'].abs().max()
            avg_TAS = slice_data['TAS'].mean() if slice_data['TAS'].mean() > 0 else avg_speed
            
            avg_speed_of_sound = calculate_isa_speed_of_sound(avg_alt) if not pd.isna(avg_alt) else 340.294 
            avg_mach = (avg_TAS * KNOT_TO_M_S) / avg_speed_of_sound if avg_speed_of_sound > 0 and avg_TAS > 0 else 0.0
            
            flight_phase = 'Cruise'
            if abs(avg_vert_rate) > 500:
                flight_phase = 'Climb' if avg_vert_rate > 0 else 'Descent'
            if avg_alt < 10000 and flight_phase == 'Cruise': flight_phase = 'Unknown'
            
        else:
            # CASE B: Blind Data (ZERO IMPUTATION FIX)
            # Reverting to 0.0 allows the model to treat this as "Static/Stopped"
            is_missing_data = 1
            avg_alt = 0.0
            avg_speed = 0.0
            avg_vert_rate = 0.0
            delta_alt = 0.0
            max_abs_vert_rate = 0.0
            avg_TAS = 0.0
            avg_mach = 0.0
            flight_phase = 'Missing'

        # Derived Features
        # NOTE: 0.0 * X = 0.0, so proxies naturally become 0 for missing data
        dist_approx = avg_speed * (duration/3600) 
        time_ratio = (t_start - takeoff_time).total_seconds() / total_flight_duration if total_flight_duration > 0 else 0
        mass_proxy = specs['MTOW_kg'] * (1.0 - time_ratio)
        
        features_list_local.append({
            'idx': row_idx, 'flight_id': flight_id, 'fuel_kg': fuel_kg,
            'aircraft_type': aircraft_type, 'flight_phase': str(flight_phase),
            'gear_config': specs['gear_config'], 'icao_wtc': specs['icao_wtc'], 'engine_class': specs['engine_class'],
            'route_distance_nm': route_distance_nm, 'elev_diff_ft': elev_diff_ft,
            'hour_sin': np.sin(2 * np.pi * t_start.hour / 24),
            'hour_cos': np.cos(2 * np.pi * t_start.hour / 24),
            'MTOW_kg': specs['MTOW_kg'], 'MALW_kg': specs['MALW_kg'], 'approach_speed_ms': specs['approach_speed_ms'],
            'tail_height_m': specs['tail_height_m'], 'wheelbase_m': specs['wheelbase_m'], 
            'wingspan_m': specs['wingspan_m'], 'length_m': specs['length_m'],
            'duration': duration, 'avg_altitude': avg_alt, 'avg_groundspeed': avg_speed, 'avg_TAS': avg_TAS,
            'avg_mach': avg_mach, 'avg_vert_rate': avg_vert_rate, 'max_abs_vert_rate': max_abs_vert_rate,
            'delta_altitude': delta_alt,
            'time_into_flight_ratio': time_ratio, 'mass_ratio_proxy': mass_proxy,
            'aspect_ratio_proxy': specs['wingspan_m'] / specs['length_m'] if (specs['wingspan_m']>0 and specs['length_m']>0) else np.nan,
            'tail_length_ratio': specs['tail_height_m'] / specs['length_m'] if (specs['tail_height_m']>0 and specs['length_m']>0) else np.nan,
            'weight_delta_ratio': (specs['MTOW_kg'] - specs['MALW_kg'])/specs['MTOW_kg'] if specs['MTOW_kg']>0 else np.nan,
            'energy_rate_index': mass_proxy * max_abs_vert_rate,
            'mass_speed_index': mass_proxy * avg_TAS,
            'distance_nm_approx': dist_approx,
            'is_missing_data': is_missing_data
        })

    return features_list_local

#%% EXECUTION FUNCTIONS
def execute_extraction(fuel_file, flightlist_file, flights_dir, output_file, dataset_name, df_specs, df_airports):
    print(f"\n=== PROCESSING {dataset_name} ===")
    start_time = time.time()
    
    # 1. Load
    df_fuel = pd.read_parquet(os.path.join(DATA_DIR, fuel_file))
    df_fl = pd.read_parquet(os.path.join(DATA_DIR, flightlist_file))
    
    # 2. Merge Static
    df_fl = pd.merge(df_fl, df_airports, left_on='origin_icao', right_on='icao', how='left').rename(columns={'lat':'origin_lat', 'lon':'origin_lon', 'elev':'origin_elev'}).drop(columns=['icao'])
    df_fl = pd.merge(df_fl, df_airports, left_on='destination_icao', right_on='icao', how='left').rename(columns={'lat':'dest_lat', 'lon':'dest_lon', 'elev':'dest_elev'}).drop(columns=['icao'])
    
    df_fl['route_distance_nm'] = df_fl.apply(lambda r: haversine(r['origin_lat'], r['origin_lon'], r['dest_lat'], r['dest_lon']), axis=1)
    df_fl['elev_diff_ft'] = df_fl['dest_elev'] - df_fl['origin_elev']
    
    df_fl = pd.merge(df_fl, df_specs, on='aircraft_type', how='left')
    
    # 3. Filter & Task Setup
    # We filter to ensure static context exists (Intersection of Fuel & Flightlist)
    merged_ids = df_fl['flight_id'].unique()
    fuel_proc = df_fuel[df_fuel['flight_id'].isin(merged_ids)].copy()
    
    input_tasks = []
    for fid in fuel_proc['flight_id'].unique():
        entry = df_fl[df_fl['flight_id'] == fid].iloc[0].copy()
        entry['traj_path'] = os.path.join(DATA_DIR, flights_dir, f"{fid}.parquet")
        input_tasks.append((fid, entry, fuel_proc[fuel_proc['flight_id'] == fid].copy()))
    
    print(f"   Flights: {len(input_tasks)} | Intervals: {len(fuel_proc)}")
    
    # 4. Parallel Execution
    with Pool(NUM_WORKERS) as pool:
        results = pool.starmap(process_single_flight, input_tasks)
        
    df_final = pd.DataFrame([item for sublist in results for item in sublist])
    
    # 5. Final Cleanup & Save
    # Reindex ensures we have exactly the columns we want, in order.
    df_final = df_final.reindex(columns=FINAL_COLS)
    
    # Fill Missing Categoricals with String 'MISSING' (CatBoost Requirement)
    for c in CAT_FEATURES_LIST:
        df_final[c] = df_final[c].fillna("MISSING").astype(str)

    df_final.to_parquet(output_file, index=False)
    print(f"   [DONE] Saved {len(df_final)} rows to {output_file} ({time.time()-start_time:.2f}s)")

#%% MAIN RUNNER
if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- LOAD STATIC DATA ---
    try:
        df_apt = pd.read_parquet(os.path.join(DATA_DIR, "apt.parquet")).rename(columns={'icao':'icao', 'latitude':'lat', 'longitude':'lon', 'elevation':'elev'})
        df_sp_raw = pd.read_excel(AIRCRAFT_SPEC_FILE, sheet_name=0)
    except: print("[FATAL] Static load error"); exit()
    
    # 1. Strip Whitespace from Headers
    df_sp_raw.columns = df_sp_raw.columns.str.strip()
    
    # 2. Clean Keys
    if 'ICAO_Code' in df_sp_raw.columns:
        df_sp_raw['ICAO_Code'] = df_sp_raw['ICAO_Code'].astype(str).str.strip()
    
    # 3. Clean Values (Wingspan)
    if 'Wingspan_ft_with_winglets_sharklets' in df_sp_raw.columns:
        df_sp_raw['wingspan_ft'] = df_sp_raw['Wingspan_ft_with_winglets_sharklets'].fillna(df_sp_raw['Wingspan_ft_without_winglets_sharklets'])
    else:
        df_sp_raw['wingspan_ft'] = np.nan

    # 4. Rename
    df_sp = df_sp_raw.rename(columns={
        'ICAO_Code': 'aircraft_type', 'Num_Engines': 'engine_count', 
        'Physical_Class_Engine': 'engine_class', 'ICAO_WTC': 'icao_wtc', 
        'Main_Gear_Config': 'gear_config'
    })
    
    # 5. Conversions (EXPLICIT - NO LOOPS)
    df_sp['MTOW_kg'] = df_sp['MTOW_lb'] * LB_TO_KG
    df_sp['MALW_kg'] = df_sp['MALW_lb'] * LB_TO_KG
    df_sp['wingspan_m'] = df_sp['wingspan_ft'] * FT_TO_M
    df_sp['length_m'] = df_sp['Length_ft'] * FT_TO_M
    df_sp['tail_height_m'] = df_sp['Tail_Height_at_OEW_ft'] * FT_TO_M
    df_sp['wheelbase_m'] = df_sp['Wheelbase_ft'] * FT_TO_M
    df_sp['approach_speed_ms'] = df_sp['Approach_Speed_knot'] * KNOT_TO_M_S
    
    # 6. Final Clean Subset
    df_sp = df_sp[[
        'aircraft_type', 'MTOW_kg', 'MALW_kg', 'wingspan_m', 'length_m', 
        'approach_speed_ms', 'tail_height_m', 'wheelbase_m', 
        'gear_config', 'icao_wtc', 'engine_class'
    ]].copy()

    # --- EXECUTE ALL PHASES ---
    execute_extraction("fuel_train.parquet", "flightlist_train.parquet", "flights_train", FILE_TRAIN, "TRAINING", df_sp, df_apt)
    execute_extraction("fuel_rank_submission.parquet", "flightlist_rank.parquet", "flights_rank", FILE_RANK, "RANKING", df_sp, df_apt)
    execute_extraction("fuel_final_submission.parquet", "flightlist_final.parquet", "flights_final", FILE_FINAL, "FINAL", df_sp, df_apt)

#%% VERIFICATION
    print("\n=== VERIFICATION (SAFETY CHECK) ===")
    for f, n in [(FILE_TRAIN, "TRAIN"), (FILE_RANK, "RANK"), (FILE_FINAL, "FINAL")]:
        try: 
            df = pd.read_parquet(f)
            print(f"\n--- {n} ---")
            print(f"Shape: {df.shape}")
            print("Head:")
            print(df.head(3))
            
            # Check content (Not just shape)
            if 'MTOW_kg' in df.columns:
                nulls_mtow = df['MTOW_kg'].isna().sum()
                print(f"   Missing MTOW: {nulls_mtow} ({nulls_mtow/len(df):.1%})")
        except Exception as e: 

            print(f"Error verifying {n}: {e}")

