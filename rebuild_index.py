"""
Reconstruye el índice FAISS usando all-MiniLM-L6-v2 (sentence-transformers).
Ejecutar UNA VEZ localmente antes de hacer push.

Uso:
    python rebuild_index.py
"""
import re
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent / "app"
PDF_DIR  = BASE_DIR / "data" / "pdfs"
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

    print("Generando embeddings con all-MiniLM-L6-v2...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))

    print(f"✅ Índice guardado en: {FAISS_DIR}")
    print("Haz: git add app/data/faiss/ && git commit -m 'rebuild FAISS' && git push")

if __name__ == "__main__":
    rebuild()
