from google.oauth2 import service_account
from googleapiclient.discovery import build
from app.core.config import settings

def get_drive_service():
    creds = service_account.Credentials.from_service_account_file(
        settings.GOOGLE_SERVICE_ACCOUNT_JSON,
        scopes=["https://www.googleapis.com/auth/drive"]
    )

    service = build("drive", "v3", credentials=creds)
    return service

def list_files(folder_id: str):
    service = get_drive_service()
    query = f"'{folder_id}' in parents and trashed = false"
    
    files = []
    page_token = None

    while True:
        results = service.files().list(
            q=query,
            pageSize=1000,
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token
        ).execute()

        files.extend(results.get("files", []))
        page_token = results.get("nextPageToken", None)
        if not page_token:
            break

    return files