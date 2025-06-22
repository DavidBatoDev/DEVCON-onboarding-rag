"""
list_drive_folder.py
Lists all files in a specific Google Drive folder.

Usage:
    # 1️⃣ put your folder ID below (or pass it as an env-var / CLI arg)
    # 2️⃣ make sure GOOGLE_SERVICE_ACCOUNT_JSON or
    #    GOOGLE_SERVICE_ACCOUNT_JSON_FILE is set
    # 3️⃣ python list_drive_folder.py
"""

import os, json, argparse, pathlib
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def load_credentials():
    """
    Supports multiple credential loading methods:
      • Individual env vars (recommended for production)
      • JSON file (for local development)
      • JSON string (fallback)
    """
    
    # Method 1: Individual environment variables (most deployment-friendly)
    project_id = os.getenv("GOOGLE_PROJECT_ID")
    private_key = os.getenv("GOOGLE_PRIVATE_KEY")
    private_key_id = os.getenv("GOOGLE_PRIVATE_KEY_ID") 
    client_email = os.getenv("GOOGLE_CLIENT_EMAIL")
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    
    if all([project_id, private_key, client_email]):
        print("✅ Loading credentials from individual environment variables")
        
        # Fix private key formatting if needed
        if private_key and not private_key.startswith("-----BEGIN PRIVATE KEY-----"):
            private_key = f"-----BEGIN PRIVATE KEY-----\n{private_key}\n-----END PRIVATE KEY-----\n"
        
        # Replace \\n with actual newlines
        private_key = private_key.replace('\\n', '\n')
        
        creds_info = {
            "type": "service_account",
            "project_id": project_id,
            "private_key_id": private_key_id or "default",
            "private_key": private_key,
            "client_email": client_email,
            "client_id": client_id or "default",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{client_email.replace('@', '%40')}",
            "universe_domain": "googleapis.com"
        }
        
        return service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
    
    # Method 2: JSON file (for local development)
    json_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_FILE")
    if json_path and os.path.exists(json_path):
        print(f"✅ Loading credentials from file: {json_path}")
        return service_account.Credentials.from_service_account_file(json_path, scopes=SCOPES)
    
    # Method 3: JSON string (fallback)
    raw_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_DATA")
    if raw_json:
        print("✅ Loading credentials from JSON environment variable")
        try:
            info = json.loads(raw_json)
            return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing JSON from environment variable: {e}")
            raise

    raise RuntimeError(
        "❌ No credentials found.\n"
        "Set individual env vars (GOOGLE_PROJECT_ID, GOOGLE_PRIVATE_KEY, etc.) or\n"
        "GOOGLE_SERVICE_ACCOUNT_JSON_FILE (path) or\n" 
        "GOOGLE_SERVICE_ACCOUNT_JSON_DATA (JSON string)."
    )

def list_files_in_folder(drive, folder_id):
    query = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    file_count = 0
    
    print(f"📁 Listing files in folder: {folder_id}")
    print("-" * 80)
    
    while True:
        try:
            response = drive.files().list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                pageToken=page_token,
                pageSize=100  # Get more files per request
            ).execute()

            files = response.get("files", [])
            if not files and file_count == 0:
                print("📂 No files found in this folder.")
                return

            for f in files:
                file_count += 1
                size = f.get('size', 'N/A')
                modified = f.get('modifiedTime', 'N/A')
                print(f"{file_count:3d}. {f['name']:<50} [{f['id']}]")
                print(f"     Type: {f['mimeType']:<40} Size: {size}")
                print()

            page_token = response.get("nextPageToken")
            if not page_token:
                break
                
        except HttpError as e:
            print(f"❌ Drive API error: {e}")
            if e.resp.status == 404:
                print("💡 The folder ID might be incorrect or you don't have access to it.")
            elif e.resp.status == 403:
                print("💡 Permission denied. Make sure your service account has access to this folder.")
            raise
    
    print(f"✅ Total files found: {file_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List Drive folder contents.")
    parser.add_argument("--folder", "-f", help="Google Drive folder ID")
    args = parser.parse_args()

    folder_id = (
        args.folder
        or os.getenv("GOOGLE_DRIVE_FOLDER_ID")
        or "YOUR_DEFAULT_FOLDER_ID_HERE"
    )

    if not folder_id or folder_id == "YOUR_DEFAULT_FOLDER_ID_HERE":
        raise SystemExit("❌ You must supply --folder or set GOOGLE_DRIVE_FOLDER_ID.")

    print("🔐 Loading Google Drive credentials...")
    
    try:
        creds = load_credentials()
        print("✅ Credentials loaded successfully")
        
        print("🔗 Building Drive service...")
        drive_service = build("drive", "v3", credentials=creds)
        print("✅ Drive service built successfully")
        
        list_files_in_folder(drive_service, folder_id)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Verify your service account JSON file exists and is valid")
        print("2. Check that the folder ID is correct")
        print("3. Ensure your service account has permission to access the folder")
        print("4. Try sharing the folder with your service account email")