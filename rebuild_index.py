"""
Reconstruye el índice FAISS usando FastEmbed (all-MiniLM-L6-v2, sin torch).
Ejecutar UNA VEZ localmente antes de hacer push.

Instalar antes: pip install fastembed langchain-community faiss-cpu pypdf langchain-text-splitters

Uso:
    python rebuild_index.py
"""
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

BASE_DIR  = Path(__file__).resolve().parent / "app"
PDF_DIR   = BASE_DIR / "data" / "pdfs"
FAISS_DIR = BASE_DIR / "data" / "faiss"

def clean_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def rebuild():
    print(f"Buscando PDFs en: {PDF_DIR}")
    documents = []

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        print(f"  Cargando: {pdf.name}")
        try:
            pages = PyPDFLoader(str(pdf)).load()
            for p in pages:
                p.page_content = clean_text(p.page_content)
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

    print("Generando embeddings con FastEmbed (all-MiniLM-L6-v2)...")
    embeddings = FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))

    print(f"✅ Índice guardado en: {FAISS_DIR}")
    print("Haz: git add app/data/faiss/ && git commit && git push")

if __name__ == "__main__":
    rebuild()
