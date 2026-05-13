"""
Reconstruye el índice FAISS con metadata de categoría por chunk.
Esto permite filtrar por categoría en la búsqueda y evitar mezcla de información.

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

def extract_category(filename: str) -> str:
    """Extrae la categoría del nombre del archivo: categoria_año-semestre_nombre.pdf"""
    parts = filename.split("_")
    if parts:
        return parts[0].lower()
    return "general"

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
                # Guardar categoría en metadata de cada página
                p.metadata["category"] = category
                p.metadata["filename"] = pdf.name
            documents.extend(pages)
        except Exception as e:
            print(f"  ⚠️ Error: {e}")

    if not documents:
        print("❌ No se encontraron PDFs en app/data/pdfs/")
        return

    print(f"\nPáginas cargadas: {len(documents)}")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(documents)
    print(f"Chunks generados: {len(chunks)}")

    # Verificar distribución por categoría
    from collections import Counter
    cats = Counter(c.metadata.get("category", "?") for c in chunks)
    print("\nDistribución de chunks por categoría:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count} chunks")

    print("\nGenerando embeddings con FastEmbed...")
    embeddings = FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))

    print(f"\n✅ Índice guardado en: {FAISS_DIR}")
    print("Haz: git add app/data/faiss/ && git commit && git push")

if __name__ == "__main__":
    rebuild()
