# backend/app/services/document_processor.py
from llama_index.core import Document
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from app.core.config import settings
from app.services.drive_loader import get_drive_service
import io
import docx2txt
import PyPDF2
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import Google Docs reader, with fallback
try:
    from llama_index.readers.google import GoogleDocsReader
except ImportError:
    try:
        from llama_index.readers.google import GoogleDocsReader
    except ImportError:
        GoogleDocsReader = None
        logger.warning("GoogleDocsReader not available. Google Docs will be processed via export.")


class EnhancedDocumentProcessor:
    def __init__(self):
        self.drive_service = get_drive_service()
        self.google_docs_reader = GoogleDocsReader() if GoogleDocsReader else None
    
    def process_drive_files(self, files: List[Dict[str, Any]]) -> List[Document]:
        """Process multiple files from Google Drive into LlamaIndex Documents"""
        documents = []
        
        for file in files:
            try:
                doc = self.process_single_file(file)
                if doc:
                    documents.append(doc)
                    logger.info(f"✅ Processed: {file.get('name', 'Unknown')}")
                else:
                    logger.warning(f"⚠️ Failed to process: {file.get('name', 'Unknown')}")
            except Exception as e:
                logger.error(f"❌ Error processing {file.get('name', 'Unknown')}: {e}")
                continue
        
        return documents
    
    def process_single_file(self, file: Dict[str, Any]) -> Document:
        """Process a single file into a LlamaIndex Document"""
        file_id = file["id"]
        file_name = file.get("name", "Unknown")
        mime_type = file["mimeType"]
        
        # Extract text based on file type
        text_content = self.extract_text_by_type(file_id, mime_type, file_name)

        title = file.get('name', 'Unknown')
        if mime_type == "application/vnd.google-apps.document":
            title = self._extract_google_doc_title(file_id)
        
        if not text_content or text_content.strip() == "":
            return None
        
        # Create enhanced metadata
        metadata = {
            "source": file_name,
            "file_id": file_id,
            "mime_type": mime_type,
            "file_size": file.get("size", "Unknown"),
            "created_time": file.get("createdTime", "Unknown"),
            "modified_time": file.get("modifiedTime", "Unknown"),
            "owners": [owner.get("displayName", "Unknown") for owner in file.get("owners", [])],
            "drive_link": f"https://drive.google.com/file/d/{file_id}/view",
            "title": title,
        }
        
        # Create LlamaIndex Document
        document = Document(
            text=text_content,
            metadata=metadata,
            id_=file_id  # Use file_id as document ID
        )
        
        return document
    
    def extract_text_by_type(self, file_id: str, mime_type: str, file_name: str) -> str:
        """Extract text based on file MIME type"""
        try:
            if mime_type == "application/vnd.google-apps.document":
                return self._extract_google_doc(file_id)
            
            elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return self._extract_docx(file_id)
            
            elif mime_type == "application/pdf":
                return self._extract_pdf(file_id)
            
            elif mime_type == "text/plain":
                return self._extract_plain_text(file_id)
            
            elif mime_type in ["application/vnd.google-apps.spreadsheet", 
                             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                # For spreadsheets, we'll extract as CSV
                return self._extract_spreadsheet(file_id, mime_type)
            
            else:
                logger.warning(f"Unsupported file type: {mime_type} for {file_name}")
                return f"[Unsupported file type: {mime_type}]"
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_name}: {e}")
            return f"[Error extracting text: {str(e)}]"
    
    def _extract_google_doc(self, file_id: str) -> str:
        """Extract text from Google Docs"""
        try:
            if self.google_docs_reader:
                # Use LlamaIndex Google Docs reader if available
                documents = self.google_docs_reader.load_data(document_ids=[file_id])
                if documents:
                    return documents[0].text
        except Exception as e:
            logger.debug(f"GoogleDocsReader failed, falling back to export: {e}")
        
        # Fallback to manual export as plain text
        try:
            exported = self.drive_service.files().export(
                fileId=file_id, 
                mimeType="text/plain"
            ).execute()
            return exported.decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to export Google Doc {file_id}: {e}")
            return f"[Error extracting Google Doc: {str(e)}]"
    
    def _extract_docx(self, file_id: str) -> str:
        """Extract text from DOCX files"""
        request = self.drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        
        # Write to temporary file and extract text
        temp_path = f"temp_{file_id}.docx"
        try:
            with open(temp_path, "wb") as f:
                f.write(fh.read())
            
            text = docx2txt.process(temp_path)
            return text
        finally:
            # Clean up temporary file
            import os
            try:
                os.remove(temp_path)
            except:
                pass
    
    def _extract_pdf(self, file_id: str) -> str:
        """Extract text from PDF files"""
        request = self.drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        
        # Extract text using PyPDF2
        try:
            pdf_reader = PyPDF2.PdfReader(fh)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF {file_id}: {e}")
            return f"[Error extracting PDF: {str(e)}]"
    
    def _extract_plain_text(self, file_id: str) -> str:
        """Extract plain text files"""
        request = self.drive_service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        
        try:
            return fh.read().decode("utf-8")
        except UnicodeDecodeError:
            # Try with different encoding
            fh.seek(0)
            return fh.read().decode("utf-8", errors="ignore")
    
    def _extract_spreadsheet(self, file_id: str, mime_type: str) -> str:
        """Extract spreadsheet data as CSV text"""
        try:
            if mime_type == "application/vnd.google-apps.spreadsheet":
                # Export Google Sheets as CSV
                exported = self.drive_service.files().export(
                    fileId=file_id, 
                    mimeType="text/csv"
                ).execute()
                return exported.decode("utf-8")
            else:
                # For Excel files, this would need additional processing
                return "[Excel spreadsheet data - processing not fully implemented]"
        except Exception as e:
            logger.error(f"Error extracting spreadsheet {file_id}: {e}")
            return f"[Error extracting spreadsheet: {str(e)}]"

    def _extract_google_doc_title(self, file_id: str) -> str:
        """Get actual title from Google Doc (not just filename)"""
        try:
            doc = self.drive_service.files().get(
                fileId=file_id, 
                fields="name"
            ).execute()
            return doc.get('name', 'Untitled Document')
        except Exception as e:
            logger.error(f"Error getting Google Doc title: {e}")
            return "Google Document"

# Global instance
document_processor = EnhancedDocumentProcessor()