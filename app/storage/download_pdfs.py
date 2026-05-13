import os
from pathlib import Path

DOWNLOAD_PATH = Path("app/data/pdfs")

def download_pdfs():
    """Descarga PDFs desde Supabase. Si falla, continúa con los que ya están en disco."""
    DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

    try:
        from app.storage.supabase_client import get_supabase
        client = get_supabase()
        files = client.storage.from_("pdfs").list(path="")
        print(f"[SUPABASE] Archivos encontrados: {len(files)}")
    except Exception as e:
        print(f"[SUPABASE] No se pudo conectar, usando PDFs locales: {e}")
        return

    for file in files:
        file_name = file.get("name")
        if not file_name:
            continue

        dest = DOWNLOAD_PATH / file_name
        if dest.exists():
            print(f"[SUPABASE] Ya existe localmente: {file_name}")
            continue

        try:
            data = client.storage.from_("pdfs").download(file_name)
            if data:
                with open(dest, "wb") as f:
                    f.write(data)
                print(f"[SUPABASE] ✅ Descargado: {file_name}")
            else:
                print(f"[SUPABASE] ❌ Sin datos: {file_name}")
        except Exception as e:
            print(f"[SUPABASE] ❌ Error descargando {file_name}: {e}")
