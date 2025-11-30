from minio import Minio
from minio.error import S3Error
import os
import re
import glob

# --- CONFIGURATION ---
# 1. Credentials
ENDPOINT = "s3.opensky-network.org"
ACCESS_KEY = ######## 
SECRET_KEY = ########

# 2. Bucket Info
TEAM_BUCKET = "prc-2025-merry-jacket"
TEAM_NAME = "merry-jacket"

# 3. Paths
SUBMISSION_DIR = "../submissions"

# --- FIND LATEST VERSION ---
def get_latest_submission_file(directory, team_name):
    search_pattern = os.path.join(directory, f"{team_name}_v*.parquet")
    files = glob.glob(search_pattern) 
    if not files:
        return None, None

    # Helper to extract version number and mtime
    def get_version_and_time(filepath):
        filename = os.path.basename(filepath)
        # Regex to find the integer after '_v'
        match = re.search(r"_v(\d+)", filename)
        if match:
            version = int(match.group(1))
        else:
            version = 0 # Fallback if naming format is weird
        
        # Get modification time for tie-breaking (latest file wins)
        mtime = os.path.getmtime(filepath)
        return (version, mtime)

    # Sort files by Version (descending) then Time (descending)
    latest_file = max(files, key=get_version_and_time)
    
    return latest_file, os.path.basename(latest_file)

# --- UPLOAD LOGIC ---
def upload_submission():
    # 1. Connect
    client = Minio(
        endpoint=ENDPOINT,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=True
    )

    print(f"[INIT] Connecting to {ENDPOINT}...")

    try:
        # 2. Auto-Select File
        print(f"[SCAN] Scanning '{SUBMISSION_DIR}' for latest submission...")
        local_path, remote_name = get_latest_submission_file(SUBMISSION_DIR, TEAM_NAME)
        
        if not local_path:
            print(f"[ERROR] No submission files found matching '{TEAM_NAME}_v*.parquet'")
            return

        print(f"   Found Latest Local File: {local_path}")
        print(f"   Target Remote Name:      {remote_name}")

        # 3. Check Bucket Permissions
        if not client.bucket_exists(TEAM_BUCKET):
            print(f"[ERROR] Bucket '{TEAM_BUCKET}' not found or permission denied.")
            return
        
        print(f"[CHECK] Bucket '{TEAM_BUCKET}' exists.")
        
        # 4. Upload
        print(f"\n[UPLOAD] Uploading '{remote_name}'...")
        
        result = client.fput_object(
            bucket_name=TEAM_BUCKET,
            object_name=remote_name, # Uploads with the exact same name as the local file
            file_path=local_path,
        )
        
        print("\n[SUCCESS] File uploaded successfully!")
        print(f"   Bucket:  {TEAM_BUCKET}")
        print(f"   File:    {remote_name}")
        print(f"   Etag:    {result.etag}")
        
    except S3Error as e:
        print(f"\n[FAILURE] S3 Error occurred: {e}")

if __name__ == "__main__":
    upload_submission()
    
#%% UPLOAD FINAL SUBMISSION

def upload_final_specific():
    # 1. Define the exact file
    FINAL_FILENAME = "merry-jacket_final.parquet"
    local_path = os.path.join(SUBMISSION_DIR, FINAL_FILENAME)
    
    print(f"\n[INIT] Final Submission Upload: {FINAL_FILENAME}")
    
    # 2. Check existence
    if not os.path.exists(local_path):
        print(f"[FATAL] File not found at: {local_path}")
        print("Please check the filename and directory.")
        return

    # 3. Connect
    client = Minio(
        endpoint=ENDPOINT,
        access_key=ACCESS_KEY,
        secret_key=SECRET_KEY,
        secure=True
    )
    
    # 4. Upload
    print(f"[UPLOAD] Sending {FINAL_FILENAME} to bucket '{TEAM_BUCKET}'...")
    try:
        result = client.fput_object(
            bucket_name=TEAM_BUCKET,
            object_name=FINAL_FILENAME, # Remote name matches local name
            file_path=local_path,
        )
        print("\n[SUCCESS] FINAL SUBMISSION UPLOADED!")
        print(f"   Bucket: {TEAM_BUCKET}")
        print(f"   File:   {FINAL_FILENAME}")
        print(f"   Etag:   {result.etag}")
        
    except S3Error as e:
        print(f"\n[FAILURE] Upload failed: {e}")

# Run it
if __name__ == "__main__":

    upload_final_specific()
