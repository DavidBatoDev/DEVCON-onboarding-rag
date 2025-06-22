# backend/app/services/drive_loader.py
from google.oauth2 import service_account
from googleapiclient.discovery import build
from app.core.config import settings
import logging
import os
import json

logger = logging.getLogger(__name__)

def get_drive_service():
    """Create and return Google Drive service client"""
    try:
        raw = settings.GOOGLE_SERVICE_ACCOUNT_JSON_DATA or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_DATA")
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]

        # If raw starts with “{”, treat it as JSON; otherwise treat it as a path
        if raw and raw.strip().startswith("{"):
            info = json.loads(raw)
            creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
        else:
            creds = service_account.Credentials.from_service_account_file(raw, scopes=scopes)
        service = build("drive", "v3", credentials=creds)
        logger.info("✅ Google Drive service initialized")
        return service
    except Exception as e:
        logger.error(f"❌ Error initializing Google Drive service: {e}")
        raise

def get_drive_files(folder_id: str = None):
    """
    Get files from Google Drive folder
    
    Args:
        folder_id: Google Drive folder ID. If None, gets files from root or configured folder
    
    Returns:
        List of file dictionaries with id, name, mimeType, and size
    """
    try:
        service = get_drive_service()
          # Build query
        if folder_id:
            query = f"'{folder_id}' in parents and trashed = false"
        elif settings.GOOGLE_DRIVE_FOLDER_ID:
            # Use default folder ID from settings
            query = f"'{settings.GOOGLE_DRIVE_FOLDER_ID}' in parents and trashed = false"
            logger.info(f"📁 Using default folder ID: {settings.GOOGLE_DRIVE_FOLDER_ID}")
        else:
            # If no folder_id provided, get files from root
            query = "trashed = false"
        
        logger.info(f"🔍 Searching Google Drive with query: {query}")
        
        files = []
        page_token = None

        while True:
            results = service.files().list(
                q=query,
                pageSize=1000,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, parents)",
                pageToken=page_token
            ).execute()

            batch_files = results.get("files", [])
            files.extend(batch_files)
            
            logger.info(f"📁 Found {len(batch_files)} files in this batch")
            
            page_token = results.get("nextPageToken", None)
            if not page_token:
                break

        logger.info(f"📋 Total files found: {len(files)}")
        
        # Filter for document types that can be processed
        supported_types = [
            'application/pdf',
            'application/vnd.google-apps.document',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/plain',
            'application/vnd.google-apps.presentation',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'application/vnd.ms-powerpoint'
        ]
        
        filtered_files = [f for f in files if f.get('mimeType') in supported_types]
        
        if len(filtered_files) != len(files):
            logger.info(f"📄 Filtered to {len(filtered_files)} supported document types")
        
        return filtered_files

    except Exception as e:
        logger.error(f"❌ Error getting files from Google Drive: {e}")
        raise

# Alias for backward compatibility
def list_files(folder_id: str):
    """Backward compatibility alias for get_drive_files"""
    return get_drive_files(folder_id)