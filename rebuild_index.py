"""
Ejecuta este script UNA VEZ en tu máquina local para reconstruir
el índice FAISS usando sentence-transformers (all-MiniLM-L6-v2).
El servidor usa HuggingFaceInferenceAPIEmbeddings con el mismo modelo,
así los vectores son compatibles.

Uso:
    python rebuild_index.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
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

    if not PDF_DIR.exists():
        print(f"❌ La carpeta {PDF_DIR} no existe.")
        return

    for pdf in PDF_DIR.glob("*.pdf"):
        print(f"  Cargando: {pdf.name}")
        try:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()
            for page in pages:
                page.page_content = clean_text(page.page_content)
            documents.extend(pages)
        except Exception as e:
            print(f"  ⚠️ Error cargando {pdf.name}: {e}")

    if not documents:
        print("❌ No se encontraron PDFs.")
        return

    print(f"Total páginas: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"Chunks generados: {len(chunks)}")

    print("Generando embeddings locales (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))

    print("-" * 30)
    print(f"✅ Índice FAISS guardado en: {FAISS_DIR}")
    print("Haz commit de app/data/faiss/ y push.")

if __name__ == "__main__":
    rebuild()

# Rutas de carpetas
BASE_DIR = Path(__file__).resolve().parent / "app"
PDF_DIR = BASE_DIR / "data" / "pdfs"
FAISS_DIR = BASE_DIR / "data" / "faiss"

def clean_text(text):
    """Limpia el texto extraído del PDF para mejorar los embeddings."""
    text = re.sub(r'-\n', '', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def rebuild():
    print(f"Buscando PDFs en: {PDF_DIR}")
    documents = []
    
    if not PDF_DIR.exists():
        print(f"❌ Error: La carpeta {PDF_DIR} no existe.")
        return

    for pdf in PDF_DIR.glob("*.pdf"):
        print(f"  Cargando: {pdf.name}")
        try:
            loader = PyPDFLoader(str(pdf))
            pages = loader.load()
            for page in pages:
                page.page_content = clean_text(page.page_content)
            documents.extend(pages)
        except Exception as e:
            print(f"  ⚠️ No se pudo cargar {pdf.name}: {e}")

    if not documents:
        print("❌ No se encontraron PDFs válidos en app/data/pdfs/")
        return

    print(f"Total páginas cargadas: {len(documents)}")

    # Dividir el texto en fragmentos (chunks)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"Chunks generados: {len(chunks)}")

    print("Configurando Google Embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # --- PROCESAMIENTO POR LOTES (BATCHING) ---
    print("Generando embeddings y construyendo índice FAISS...")
    
    # 1. Inicializamos el vectorstore con el PRIMER chunk para crear el objeto correctamente
    # Usamos un solo documento para evitar que la API agrupe respuestas inesperadamente
    vectorstore = FAISS.from_documents(chunks[:1], embeddings)
    print(f"  Progreso: 1/{len(chunks)} fragmentos procesados...")

    # 2. Procesamos el resto de los chunks UNO POR UNO
    # Aunque es un poco más lento, garantiza que no haya error de longitud (mismatch)
    for i in range(1, len(chunks)):
        try:
            batch = [chunks[i]]
            vectorstore.add_documents(batch)
            if (i + 1) % 10 == 0: # Imprimimos progreso cada 10 para no saturar la consola
                print(f"  Progreso: {i + 1}/{len(chunks)} fragmentos procesados...")
            
            # Un pequeño respiro para la API gratuita
            time.sleep(0.5) 
        except Exception as e:
            print(f"  ⚠️ Error en chunk {i}: {e}")
            continue

    # 3. Guardar el resultado
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))
    
    print("-" * 30)
    print(f"✅ ¡ÉXITO! Índice FAISS guardado en: {FAISS_DIR}")
    print("Ahora puedes hacer commit de la carpeta 'app/data/faiss/' y subirla a GitHub.")

if __name__ == "__main__":
    rebuild()