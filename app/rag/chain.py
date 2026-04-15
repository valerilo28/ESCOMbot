import google.generativeai as genai
import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from app.rag.loader import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import time

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_DIR = BASE_DIR / "data" / "pdfs"
FAISS_DIR = BASE_DIR / "data" / "faiss"

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

last_call_time = 0

def can_call_model():
    global last_call_time
    now = time.time()

    if now - last_call_time < 2:
        return False

    last_call_time = now
    return True


def is_good_context(docs):
    if not docs:
        return False

    total_length = sum(len(d.page_content) for d in docs)
    return total_length > 500


def load_chain():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_for_bm25 = splitter.split_documents(documents)

    bm25_retriever = BM25Retriever.from_documents(docs_for_bm25)
    bm25_retriever.k = 2

    model = genai.GenerativeModel("gemini-1.5-flash")

    cache = {}
    chat_history = []

    def chain(question: str):

        if question in cache:
            print("[LOG] Cache hit")
            return cache[question]

        docs_faiss = faiss_retriever.invoke(question)
        docs_bm25 = bm25_retriever.invoke(question)

        all_docs = docs_faiss + docs_bm25

        unique_contents = set()
        final_docs = []
        for d in all_docs:
            if d.page_content not in unique_contents:
                final_docs.append(d)
                unique_contents.add(d.page_content)

        context = "\n".join(d.page_content for d in final_docs[:2])

        if is_good_context(final_docs):
            print("[LOG] Sin IA")
            answer = f"Según documentos oficiales:\n\n{context[:1000]}"
            chat_history.append((question, answer))
            cache[question] = answer
            return answer

        if not can_call_model():
            return "Espera un momento antes de hacer otra pregunta"

        print("[LOG] Usando Gemini")

        history_text = "\n".join(
            [f"Usuario: {q}\nBot: {a}" for q, a in chat_history[-3:]]
        )

        prompt = f"""
Eres un asistente oficial de ESCOM IPN.

Historial:
{history_text}

Contexto:
{context}

Pregunta:
{question}

Reglas:
- Usa el contexto, sin mencionar que estás contestando bajo un contexto.
- Responde de forma natural, que no se vea tan robotizado si armado.
- Trata de dar respuestas cortas y concisas, que respondan claramente lo que se te está preguntando.
- Mantén coherencia con historial, es decir, trata de recordar lo que se te fue preguntando anteriormente para que sepas de lo que se te está preguntando.
- No inventes cosas fuera de contexto.
"""

        response = model.generate_content(prompt)
        answer = response.text

        chat_history.append((question, answer))
        cache[question] = answer

        return answer

    return chain