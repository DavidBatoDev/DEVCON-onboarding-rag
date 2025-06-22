from app.services.drive_loader import list_files
from app.services.extractor import extract_text
from app.services.embedder import embed_texts
from app.services.vector_store import save_to_store
from app.core.config import settings

# Replace with your actual Google Drive folder ID
FOLDER_ID = "1eocL8T8BH6EwnP5siOtDz3FG2CqGHveS"

print("Starting full document compilation...")

files = list_files(FOLDER_ID)
print(f"Found {len(files)} files in Drive folder.")

texts = []
metadatas = []

for file in files:
    print(f"\nProcessing: {file['name']} ({file['id']})")
    try:
        content = extract_text(file)
        if content.startswith("[Unsupported"):
            print(f"Skipped unsupported file: {file['mimeType']}")
            continue

        texts.append(content)
        metadatas.append({
            "file_id": file["id"],
            "file_name": file["name"],
            "content": content  # Add this line
        })
        print(f"Extracted {len(content)} characters.")
    except Exception as e:
        print(f"Failed to process {file['name']}: {e}")

print("Embedding and saving...")
vectors = embed_texts(texts)
save_to_store(metadatas, vectors, settings.RAG_INDEX_DIR)
print("✅ Compilation complete.")