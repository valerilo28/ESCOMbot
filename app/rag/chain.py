import google.generativeai as genai
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from app.rag.loader import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent

print(BASE_DIR)

PDF_DIR = BASE_DIR / "data" / "pdfs"
FAISS_DIR = BASE_DIR / "data" / "faiss"

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def load_chain():
    # --- EMBEDDINGS Y VECTORSTORE ---

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_for_bm25 = splitter.split_documents(documents)
    
    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = 3

    # --- MODELO ---
    #model = genai.GenerativeModel("models/gemini-2.5-flash")
    model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

    def chain(question: str):
        # 1. Búsqueda Semántica
        docs_faiss = faiss_retriever.invoke(question)
        
        # 2. Búsqueda por Palabras Clave
        docs_bm25 = bm25_retriever.invoke(question)
        
        all_docs = docs_faiss + docs_bm25
        
        unique_contents = set()
        final_docs = []
        for d in all_docs:
            if d.page_content not in unique_contents:
                final_docs.append(d)
                unique_contents.add(d.page_content)

        print(f"\n[LOG] Pregunta: {question}")
        print(f"[LOG] Híbrido: FAISS ({len(docs_faiss)}) + BM25 ({len(docs_bm25)}) -> Total Únicos: {len(final_docs)}")
        
        context = "\n".join(d.page_content for d in final_docs)

        prompt = f"""
Eres un asistente oficial de la Escuela Superior de Cómputo (ESCOM) del IPN.

Reglas estrictas:
- Responde únicamente con la información proporcionada en el contexto.
- No inventes información.
- Si el contexto no contiene la respuesta solicitada, responde exactamente:
- Si la respuesta no está en el contexto, responde:
  "No se encontró información oficial de ESCOM sobre esa pregunta."

Contexto:
{context}

Pregunta:
{question}
"""

        response = model.generate_content(prompt)
        return response.text

    return chain
