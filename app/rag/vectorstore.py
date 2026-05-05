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
    text = re.sub(r'-\n', '', text)        # une palabras cortadas con guión
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # une líneas sueltas
    text = re.sub(r'\s{2,}', ' ', text)    # colapsa espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def build_vectorstore():
    documents = []
    embeddings = HuggingFaceEmbeddings(  # ✅ Definir PRIMERO
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    for pdf in PDF_DIR.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        for page in pages:
            page.page_content = clean_text(page.page_content)
        documents.extend(pages)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
    )
    chunks = text_splitter.split_documents(documents)  # ✅ Todos los docs

    vectorstore = FAISS.from_documents(chunks, embeddings)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))

    print("Vectorstore FAISS creado correctamente")

if __name__ == "__main__":
    build_vectorstore()
