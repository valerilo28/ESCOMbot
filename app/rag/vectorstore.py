from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import re

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "data" / "pdfs"
FAISS_DIR = BASE_DIR / "data" / "faiss"

print(BASE_DIR)

def clean_text(text):
    text = re.sub(r'(?<=[a-zA-ZáéíóúÁÉÍÓÚ]) (?=[a-zA-ZáéíóúÁÉÍÓÚ] )', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_vectorstore():
    documents = []

    print(" Buscando PDFs en:", PDF_DIR)

    if not PDF_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta {PDF_DIR}")


    for pdf in PDF_DIR.glob("*.pdf"):
        print(f"Cargando: {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()

        for page in pages:

            page.page_content = clean_text(page.page_content)
            
        documents.extend(pages)

    if not documents:
        raise ValueError(
            "No se cargaron documentos. "
            "Verifica que los PDFs tengan texto."
        )

        # En tu lógica de creación de fragmentos:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, # Aumentado para dar más contexto por pedazo
        chunk_overlap=300,
        add_start_index=True # Útil para trazabilidad
    )

    # Al cargar los documentos, el 'source' se guarda automáticamente en metadata
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    vector_db = FAISS.from_documents(chunks, embeddings)

    for i, doc in enumerate(docs):
        print(f"\n--- Chunk {i} ---\n{doc.page_content}")

    if not docs:
        raise ValueError("El splitter no generó chunks")

    print(f"Chunks generados: {len(docs)}")

    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    vectorstore = FAISS.from_documents(docs, embeddings)

    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))

    print("Vectorstore FAISS creado correctamente")

if __name__ == "__main__":
    build_vectorstore()
