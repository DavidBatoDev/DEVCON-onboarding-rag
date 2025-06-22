from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from app.core.config import settings
from .drive_loader import get_drive_service
import io
import docx2txt

def extract_text(file: dict) -> str:
    service = get_drive_service()
    file_id = file["id"]
    mime_type = file["mimeType"]

    if mime_type == "application/vnd.google-apps.document":
        # Export Google Doc as plain text
        exported = service.files().export(fileId=file_id, mimeType="text/plain").execute()
        return exported.decode("utf-8")

    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Download .docx file and convert
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)

        with open("temp.docx", "wb") as f:
            f.write(fh.read())

        text = docx2txt.process("temp.docx")
        return text

    else:
        return f"[Unsupported file type: {mime_type}]"