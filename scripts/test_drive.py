from app.services.drive_loader import list_files
from app.services.extractor import extract_text

FOLDER_ID = "1eocL8T8BH6EwnP5siOtDz3FG2CqGHveS"  
files = list_files(FOLDER_ID, max_results=100)

print(f"\n🔎 Accessible shared drives: []")
print(f"Found {len(files)} total files the service account can access:")

for f in files:
    name = f["name"]
    id = f["id"]
    mime = f["mimeType"]
    parents = f.get("parents", None)

    print(f"{name} ({id}) - {mime} - parents: {parents}")

# 🧪 Extract text from each file
for f in files:
    print(f"\n📄 Extracting: {f['name']}")
    content = extract_text(f)
    print(f"--- First 500 characters of extracted content ---\n{content[:500]}\n...")
