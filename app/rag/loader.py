from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import re

try:
    from app.storage.download_pdfs import download_pdfs
except ImportError:
    from storage.download_pdfs import download_pdfs

BASE_DIR = Path(__file__).resolve().parent.parent  # app/
PDF_DIR = BASE_DIR / "data" / "pdfs"

def clean_text(text):
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(?<![.:!?;])\n(?=[a-z])', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def extract_category(filename: str) -> str:
    """Extrae la categoría del nombre del archivo.
    
    Ejemplos:
      servicio_social_2026-1_Requisitos.pdf  → servicio_social
      estancia_profesional_2026-1_acred.pdf  → estancia_profesional
      becas_2026-1_convocatoria.pdf          → becas
      temario_ISC_calculo.pdf                → temario
    """
    name = filename.lower()
    if name.startswith("servicio_social"):
        return "servicio_social"
    elif name.startswith("estancia_profesional") or name.startswith("estancia"):
        return "estancia_profesional"
    elif name.startswith("beca"):
        return "becas"
    elif name.startswith("temario"):
        return "temario"
    else:
        # Fallback: tomar el primer segmento antes de "_"
        parts = name.split("_")
        return parts[0] if parts else "general"

def load_documents():
    download_pdfs()

    docs = []
    print("Buscando PDFs en:", PDF_DIR)

    if not PDF_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta {PDF_DIR}")

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        category = extract_category(pdf.name)
        print(f"Cargando [{category}]: {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        for page in pages:
            page.page_content = clean_text(page.page_content)
            # ← CRÍTICO: asignar categoría y filename en metadata
            page.metadata["category"] = category
            page.metadata["filename"] = pdf.name
            print(f"  Página {page.metadata.get('page','')} — {len(page.page_content)} chars — cat={category}")

        docs.extend(pages)

    if not docs:
        raise ValueError("No se cargó texto desde los PDFs")

    print(f"Documentos cargados: {len(docs)}")
    return docs