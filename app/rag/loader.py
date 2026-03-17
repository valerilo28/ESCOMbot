from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import re
from app.storage.download_pdfs import download_pdfs

BASE_DIR = Path(__file__).resolve().parent.parent  # app/
PDF_DIR = BASE_DIR / "data" / "pdfs"

def clean_text(text):
    text = re.sub(r'(?<=[a-zA-ZáéíóúÁÉÍÓÚ]) (?=[a-zA-ZáéíóúÁÉÍÓÚ] )', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_documents():

    download_pdfs()

    docs = []

    print("Buscando PDFs en:", PDF_DIR)

    if not PDF_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta {PDF_DIR}")

    for pdf in PDF_DIR.glob("*.pdf"):
        print(f"Cargando: {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        for page in pages:

            page.page_content = clean_text(page.page_content)
            
        docs.extend(pages)

    if not docs:
        raise ValueError("No se cargó texto desde los PDFs")

    print(f"Documentos cargados: {len(docs)}")
    return docs
