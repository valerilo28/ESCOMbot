"""
Ejecuta este script UNA VEZ en tu máquina local para reconstruir
el índice FAISS usando Google Embeddings (sin torch).

Uso:
    python rebuild_index.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import re

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent / "app"
PDF_DIR = BASE_DIR / "data" / "pdfs"
FAISS_DIR = BASE_DIR / "data" / "faiss"

def clean_text(text):
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def rebuild():
    print(f"Buscando PDFs en: {PDF_DIR}")
    documents = []
    for pdf in PDF_DIR.glob("*.pdf"):
        print(f"  Cargando: {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        for page in pages:
            page.page_content = clean_text(page.page_content)
        documents.extend(pages)

    if not documents:
        print("❌ No se encontraron PDFs. Asegúrate de tener PDFs en app/data/pdfs/")
        return

    print(f"Total páginas cargadas: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = splitter.split_documents(documents)
    print(f"Chunks generados: {len(chunks)}")

    print("Generando embeddings con Google...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))
    print(f"✅ Índice FAISS guardado en: {FAISS_DIR}")
    print("Ahora haz commit de app/data/faiss/ y push.")

if __name__ == "__main__":
    rebuild()
