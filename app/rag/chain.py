import os
import time
import datetime
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
import json
import re

try:
    from app.rag.loader import load_documents
except ImportError:
    from rag.loader import load_documents

# --- RUTAS ---
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "data" / "faiss"
FAST_RESPONSES_PATH = BASE_DIR / "rag" / "fast_responses.json"

try:
    with open(FAST_RESPONSES_PATH, "r", encoding="utf-8") as f:
        FAST_RESPONSES = json.load(f)
except FileNotFoundError:
    print(f"[WARN] fast_responses.json no encontrado")
    FAST_RESPONSES = []

_response_cache: dict = {}
last_call_time = 0

def fast_response(question: str):
    q = question.lower()
    for item in FAST_RESPONSES:
        for keyword in item["keywords"]:
            if keyword in q:
                return item["answer"]
    return None

def fix_incomplete_answer(answer: str):
    answer = answer.strip()
    if not answer.endswith((".", "!", "?")):
        sentences = re.split(r'(?<=[.!?]) +', answer)
        if len(sentences) > 1:
            return " ".join(sentences[:-1])
        return answer + "..."
    return answer

def can_call_model():
    global last_call_time
    now = time.time()
    if now - last_call_time < 1:
        return False
    last_call_time = now
    return True

def get_current_period():
    now = datetime.datetime.now()
    semester = "2" if 2 <= now.month <= 7 else "1"
    return f"{now.year}-{semester}"

def classify_question(question: str) -> str:
    q = question.lower()
    if "beca" in q:
        return "becas"
    elif "servicio" in q:
        return "servicio_social"
    elif "estancia" in q:
        return "estancia_profesional"
    return ""

def load_chain():
    print(f"[CHAIN] BASE_DIR: {BASE_DIR}")
    print(f"[CHAIN] Buscando FAISS en: {FAISS_DIR}")
    print(f"[CHAIN] FAISS existe: {FAISS_DIR.exists()}")

    # 1. Embeddings via HuggingFace Inference API — sin torch, sin memoria pesada
    hf_token = os.getenv("HF_TOKEN")  # token gratuito de huggingface.co/settings/tokens
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token or "",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("[CHAIN] Embeddings listos.")

    # 2. Cargar FAISS
    try:
        vectorstore = FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("[CHAIN] ✅ FAISS cargado correctamente.")
    except Exception as e:
        print(f"[CHAIN] ❌ Error cargando FAISS: {e}")
        return None

    # 3. Validar Groq y crear LLM
    groq_key = os.getenv("GROQ_API_KEY")
    print(f"[CHAIN] GROQ_API_KEY presente: {bool(groq_key)}")
    if not groq_key:
        print("[CHAIN] ❌ GROQ_API_KEY no configurada")
        return None

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=400,
        api_key=groq_key
    )
    print("[CHAIN] ✅ LLM Groq listo.")

    # 4. Función interna de la chain
    def chain(question: str, history_from_app: list = None):
        if not can_call_model():
            return "Por favor, espera un momento antes de enviar otra pregunta."

        cache_key = question.lower().strip()
        if cache_key in _response_cache:
            return _response_cache[cache_key]

        fast = fast_response(question)
        if fast:
            return fast

        try:
            fecha_actual = datetime.datetime.now().strftime("%B %Y")
            periodo_actual = get_current_period()

            docs = retriever.invoke(question)
            context_docs = docs

            category_filter = classify_question(question)
            if category_filter:
                filtered = [
                    d for d in docs
                    if category_filter in d.metadata.get("source", "").lower()
                ]
                if filtered:
                    context_docs = filtered

            context_text = "\n---\n".join(
                d.page_content for d in context_docs[:3]
            )

            history_text = ""
            if history_from_app:
                history_text = "\n".join([
                    f"Usuario: {h['user']}\nBot: {h['bot']}"
                    for h in history_from_app[-3:]
                ])

            prompt = f"""Eres ESCOMbot, asistente oficial de ESCOM IPN. Responde SOLO sobre becas, servicio social y estancia profesional.

FECHA ACTUAL: {fecha_actual} | CICLO: {periodo_actual}

═══ REGLAS QUE NUNCA PUEDES ROMPER ═══
REGLA 1 — SOLO USA EL CONTEXTO: Si la respuesta no está en el CONTEXTO, di exactamente: "No tengo esa información. Acude a Gestión Escolar (Edificio 1, Planta Baja) o llama al 57296000 ext. 52001."
REGLA 2 — FORMATO: Empieza con resumen en **negrita**, usa viñetas (•), resalta fechas con ⚠️, máximo 5 puntos y 100 palabras.
REGLA 3 — TONO: Directo. Sin "Claro que sí", sin relleno.
REGLA 4 — VIGENCIA: Si el documento es de otro ciclo, inicia con "⚠️ Dato de periodo anterior:"
REGLA 5 — FUERA DE TEMA: Responde "Solo puedo ayudarte con becas, servicio social y estancia profesional."

═══ CONTEXTO ═══
{context_text}

═══ HISTORIAL ═══
{history_text}

═══ PREGUNTA ═══
{question}

RESPUESTA (máximo 100 palabras):"""

            response = llm.invoke(prompt)
            answer = fix_incomplete_answer(response.content)
            _response_cache[cache_key] = answer
            return answer

        except Exception as e:
            import traceback
            print(f"[ERROR DETALLADO]\n{traceback.format_exc()}")
            print(f"[GROQ_API_KEY presente]: {bool(os.getenv('GROQ_API_KEY'))}")
            return "Lo siento, hubo un error técnico al procesar la pregunta."

    return chain