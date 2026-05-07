import os
import time
import datetime
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
import json
import re

# Importamos tu cargador de documentos
try:
    from app.rag.loader import load_documents
except ImportError:
    from rag.loader import load_documents

# Rutas de carpetas
BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "data" / "faiss"
FAST_RESPONSES_PATH = BASE_DIR / "rag" / "fast_responses.json"

try:
    with open(FAST_RESPONSES_PATH, "r", encoding="utf-8") as f:
        FAST_RESPONSES = json.load(f)
except FileNotFoundError:
    print(f"[WARN] fast_responses.json no encontrado en {FAST_RESPONSES_PATH}")
    FAST_RESPONSES = []

# --- CACHE A NIVEL DE MÓDULO (persiste entre llamadas) ---
_response_cache: dict = {}

# --- CONTROL DE RATE LIMIT ---
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
    # Semestre 2: Febrero a Julio | Semestre 1: Agosto a Enero
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

    # Embeddings via API de HuggingFace — sin torch, sin memoria pesada
    from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_TOKEN", "hf_placeholder"),
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
        print("[CHAIN] ✅ Base de datos FAISS cargada correctamente.")
    except Exception as e:
        print(f"[CHAIN] ❌ Error cargando FAISS: {e}")
        return None

    # 3. Modelo Gemini
    api_key = os.getenv("GOOGLE_API_KEY")
    print(f"[CHAIN] GOOGLE_API_KEY presente: {bool(api_key)}")

    groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            print("[ERROR] GROQ_API_KEY no está configurada en las variables de entorno")
            return None
            
    llm = ChatGroq(
    model="mistral-saba-24b",
    temperature=0.0,
    max_tokens=400,
    api_key=os.getenv("GROQ_API_KEY")
    )

    def chain(question: str, history_from_app: list = None):
        if not can_call_model():
            return "Por favor, espera un momento antes de enviar otra pregunta."

        cache_key = question.lower().strip()

        # Revisar cache
        if cache_key in _response_cache:
            return _response_cache[cache_key]

        # Respuesta rápida por keywords
        fast = fast_response(question)
        if fast:
            return fast

        try:
            fecha_actual = datetime.datetime.now().strftime("%B %Y")
            periodo_actual = get_current_period()

            # A. Recuperar documentos
            docs = retriever.invoke(question)
            context_docs = docs

            # B. Filtrar por categoría si aplica
            category_filter = classify_question(question)
            if category_filter:
                filtered = [
                    d for d in docs
                    if category_filter in d.metadata.get("source", "").lower()
                ]
                if filtered:
                    context_docs = filtered

            context_text = "\n---\n".join(d.page_content for d in context_docs[:3])

            # C. Historial reciente
            history_text = ""
            if history_from_app:
                history_text = "\n".join(
                    [f"Usuario: {h['user']}\nBot: {h['bot']}" for h in history_from_app[-3:]]
                )

            # D. Prompt
            prompt = f"""Eres ESCOMbot, asistente oficial de ESCOM IPN. Responde SOLO sobre becas, servicio social y estancia profesional.

FECHA ACTUAL: {fecha_actual} | CICLO: {periodo_actual}

═══ REGLAS QUE NUNCA PUEDES ROMPER ═══
REGLA 1 — SOLO USA EL CONTEXTO: Si la respuesta no está literalmente en el CONTEXTO, responde exactamente: "No tengo esa información en este momento. Te recomiendo acudir a Gestión Escolar (Edificio 1, Planta Baja) o llamar al 57296000 ext. 52001."

REGLA 2 — FORMATO OBLIGATORIO: Usa SIEMPRE esta estructura:
- Empieza con una línea resumen en negrita
- Usa viñetas (•) para listas, nunca guiones
- Resalta fechas con ⚠️
- Máximo 5 puntos, máximo 100 palabras en total

REGLA 3 — TONO: Directo, sin "Claro que sí", sin "Por supuesto", sin relleno.

REGLA 4 — VIGENCIA: Si el documento es de un ciclo distinto a {periodo_actual}, inicia con: "⚠️ Dato de periodo anterior — verifica vigencia:"

REGLA 5 — PREGUNTAS FUERA DE TEMA: Si preguntan sobre calificaciones, horarios, profesores u otros temas, responde: "Solo puedo ayudarte con becas, servicio social y estancia profesional. Para otros trámites, contacta a Gestión Escolar."

═══ CONTEXTO DE DOCUMENTOS ═══
{context_text}

═══ HISTORIAL RECIENTE ═══
{history_text}

═══ PREGUNTA ═══
{question}

RESPUESTA (sigue el formato, máximo 100 palabras):"""

            # E. Ejecutar
            response = llm.invoke(prompt)
            answer = fix_incomplete_answer(response.content)

            # Guardar en cache
            _response_cache[cache_key] = answer
            return answer

        # chain.py — reemplaza el except al final de la función chain()
        except Exception as e:
            import traceback
            print(f"[ERROR DETALLADO]\n{traceback.format_exc()}")
            print(f"[GROQ_API_KEY presente]: {bool(os.getenv('GROQ_API_KEY'))}")
            return "Lo siento, hubo un error técnico al procesar la pregunta."

    return chain
