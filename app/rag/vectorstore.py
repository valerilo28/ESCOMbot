from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import re

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "data" / "pdfs"
FAISS_DIR = BASE_DIR / "data" / "faiss"

print(BASE_DIR)

def clean_text(text):
    text = re.sub(r'-\n', '', text)        # une palabras cortadas con guión
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # une líneas sueltas
    text = re.sub(r'\s{2,}', ' ', text)    # colapsa espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# vectorstore.py — reemplaza build_vectorstore completo
def build_vectorstore():
    try:
        from app.rag.loader import load_documents
    except ImportError:
        from rag.loader import load_documents

    print("[VECTORSTORE] Cargando documentos...")
    documents = load_documents()

    if not documents:
        raise ValueError("No hay documentos para indexar")

    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import os
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    print(f"[VECTORSTORE] Chunks generados: {len(chunks)}")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))
    print("[VECTORSTORE] FAISS guardado correctamente")