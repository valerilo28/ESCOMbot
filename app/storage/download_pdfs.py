import os
from app.storage.supabase_client import supabase

DOWNLOAD_PATH = "app/data/pdfs"

def download_pdfs():
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    files = supabase.storage.from_("pdfs").list(path="")

    print("Archivos en Supabase:", files)

    for file in files:
        file_name = file["name"]

        data = supabase.storage.from_("pdfs").download(file_name)

        if not data:
            print(f"❌ Error descargando {file_name}")
            continue

        print(f"✅ Tamaño de {file_name}: {len(data)} bytes")

        with open(DOWNLOAD_PATH / file_name, "wb") as f:
            f.write(data)