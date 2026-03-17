import os
from app.storage.supabase_client import supabase

DOWNLOAD_PATH = "app/data/pdfs"

def download_pdfs():
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    files = supabase.storage.from_("pdfs").list()

    for file in files:
        file_name = file["name"]
        file_path = os.path.join(DOWNLOAD_PATH, file_name)

        if not os.path.exists(file_path):
            print(f"Descargando {file_name}...")

            data = supabase.storage.from_("pdfs").download(file_name)

            with open(file_path, "wb") as f:
                f.write(data)

    print("PDFs listos ✅")