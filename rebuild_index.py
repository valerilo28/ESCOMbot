"""
Reconstruye el índice FAISS localmente y lo sube a Supabase.
Ejecutar cada vez que agregues PDFs nuevos.

Uso:
    python rebuild_index.py

Flujo:
    1. Lee PDFs de app/data/pdfs/
    2. Genera embeddings con FastEmbed (all-MiniLM-L6-v2)
    3. Guarda el índice en app/data/faiss/
    4. Sube index.faiss e index.pkl a Supabase Storage (bucket: faiss-index)
    5. Haz git add app/data/faiss/ && git push para que Render lo use
"""
import re
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

BASE_DIR  = Path(__file__).resolve().parent / "app"
PDF_DIR   = BASE_DIR / "data" / "pdfs"
FAISS_DIR = BASE_DIR / "data" / "faiss"

def clean_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def extract_category(filename: str) -> str:
    parts = filename.split("_")
    return parts[0].lower() if parts else "general"

def rebuild():
    print(f"Buscando PDFs en: {PDF_DIR}")
    documents = []

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        category = extract_category(pdf.name)
        print(f"  [{category}] {pdf.name}")
        try:
            pages = PyPDFLoader(str(pdf)).load()
            for p in pages:
                p.page_content = clean_text(p.page_content)
                p.metadata["category"] = category
                p.metadata["filename"] = pdf.name
            documents.extend(pages)
        except Exception as e:
            print(f"  ⚠️ Error: {e}")

    if not documents:
        print("❌ No se encontraron PDFs en app/data/pdfs/")
        return

    print(f"Páginas cargadas: {len(documents)}")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(documents)
    print(f"Chunks generados: {len(chunks)}")

    from collections import Counter
    cats = Counter(c.metadata.get("category", "?") for c in chunks)
    print("Distribución por categoría:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count} chunks")

    print("Generando embeddings con Cohere...")
    from langchain_cohere import CohereEmbeddings
    embeddings = CohereEmbeddings(
        model="embed-multilingual-light-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))
    print(f"✅ Índice guardado en: {FAISS_DIR}")

    # Subir índice a Supabase para que Render lo descargue
    _upload_index_to_supabase()

    print("\n📌 Siguiente paso:")
    print("   git add app/data/faiss/ && git commit -m 'rebuild FAISS' && git push")

def _upload_index_to_supabase():
    try:
        from supabase import create_client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            print("[SUPABASE] Variables no configuradas, saltando upload.")
            return

        client = create_client(url, key)
        bucket = "faiss-index"

        for fname in ["index.faiss", "index.pkl"]:
            fpath = FAISS_DIR / fname
            if not fpath.exists():
                continue
            with open(fpath, "rb") as f:
                data = f.read()
            try:
                # Intentar actualizar primero, si no existe subir nuevo
                client.storage.from_(bucket).update(fname, data)
                print(f"[SUPABASE] ✅ Actualizado: {fname}")
            except Exception:
                try:
                    client.storage.from_(bucket).upload(fname, data)
                    print(f"[SUPABASE] ✅ Subido: {fname}")
                except Exception as e:
                    print(f"[SUPABASE] ⚠️ No se pudo subir {fname}: {e}")
    except Exception as e:
        print(f"[SUPABASE] ⚠️ Error conectando: {e}")
        print("El índice se guardó localmente. Haz git push para que Render lo use.")

if __name__ == "__main__":
    rebuild()
