from supabase import create_client
import os
from pathlib import Path

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "data" / "pdfs"

def download_pdfs():

    os.makedirs(PDF_DIR, exist_ok=True)

    files = supabase.storage.from_("pdfs").list()

    for file in files:

        local_file = PDF_DIR / file["name"]

        if not local_file.exists():

            data = supabase.storage.from_("pdfs").download(file["name"])

            with open(local_file, "wb") as f:
                f.write(data)

            print(f"Descargado {file['name']}")