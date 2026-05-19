import os
import time
import datetime
from pathlib import Path
from langchain_community.vectorstores import FAISS
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

def fix_incomplete_answer(answer: str) -> str:
    """Limpia y estructura la respuesta antes de enviarla."""
    answer = answer.strip()

    # Quitar markdown de código si el modelo lo mete
    answer = re.sub(r'```[\w]*\n?', '', answer).strip()

    # Normalizar viñetas — convertir -, *, • a •
    answer = re.sub(r'^[\-\*]\s+', '• ', answer, flags=re.MULTILINE)

    # Asegurar salto de línea antes de cada viñeta
    answer = re.sub(r'(?<!\n)(• )', r'\n\1', answer)

    # Colapsar más de 2 saltos de línea seguidos
    answer = re.sub(r'\n{3,}', '\n\n', answer)

    # Si termina incompleto, cortar en la última oración completa
    if not answer.endswith((".", "!", "?", ":")):
        sentences = re.split(r'(?<=[.!?]) +', answer)
        if len(sentences) > 1:
            answer = " ".join(sentences[:-1])
        else:
            answer = answer + "..."

    return answer.strip()

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

CONTINUATION_TRIGGERS = {
    "más", "mas", "más información", "más detalles", "continúa", "continua",
    "sigue", "y qué más", "que más", "dime más", "amplía", "amplia",
    "explica más", "detalla", "cuéntame más", "otro", "otra", "siguiente"
}

def is_continuation(question: str) -> bool:
    return question.lower().strip() in CONTINUATION_TRIGGERS

def expand_question(question: str, history: list) -> str:
    """Si la pregunta es una continuación, la reformula con contexto del historial."""
    if not is_continuation(question) or not history:
        return question
    # Tomar la última pregunta del usuario como contexto
    last_user_q = history[-1].get("user", "") if history else ""
    if last_user_q:
        return f"{last_user_q} — proporciona más detalles o información adicional"
    return question
    q = question.lower()
    if "beca" in q:
        return "becas"
    elif "servicio" in q:
        return "servicio_social"
    elif "estancia" in q:
        return "estancia_profesional"
    elif any(w in q for w in ["temario", "bibliograf", "materia", "unidad de aprendizaje", "libro", "autor"]):
        return "temario"
    return ""

def load_chain():
    print(f"[CHAIN] BASE_DIR: {BASE_DIR}")
    print(f"[CHAIN] Buscando FAISS en: {FAISS_DIR}")
    print(f"[CHAIN] FAISS existe: {FAISS_DIR.exists()}")

    # Cohere Embeddings — API externa, sin modelo local, sin memoria pesada
    from langchain_cohere import CohereEmbeddings
    cohere_key = os.getenv("COHERE_API_KEY")
    print(f"[CHAIN] COHERE_API_KEY presente: {bool(cohere_key)}")
    embeddings = CohereEmbeddings(
        model="embed-multilingual-light-v3.0",  # modelo ligero, soporta español
        cohere_api_key=cohere_key
    )
    print("[CHAIN] Embeddings listos.")

    # 2. Cargar FAISS
    try:
        vectorstore = FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
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

    def _get_docs(question: str, category: str = None):
        """Recupera documentos filtrando por categoría en metadata."""
        if category:
            # Buscar solo dentro de la categoría indicada
            retriever = vectorstore.as_retriever(
                search_kwargs={
                    "k": 4,
                    "filter": {"category": category}
                }
            )
        else:
            # Sin filtro — detectar categoría automáticamente
            detected = classify_question(question)
            if detected:
                retriever = vectorstore.as_retriever(
                    search_kwargs={
                        "k": 4,
                        "filter": {"category": detected}
                    }
                )
            else:
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        docs = retriever.invoke(question)
        return docs

    # 4. Función interna de la chain
    def chain(question: str, history_from_app: list = None, force_category: str = None):
        if not can_call_model():
            return "Por favor, espera un momento antes de enviar otra pregunta."

        # Expandir preguntas de continuación antes de cachear
        expanded = expand_question(question, history_from_app or [])

        # No cachear preguntas de continuación — siempre buscar más info
        cache_key = f"{force_category or ''}:{expanded.lower().strip()}"
        if not is_continuation(question) and cache_key in _response_cache:
            return _response_cache[cache_key]

        fast = fast_response(question)
        if fast:
            return fast

        try:
            fecha_actual = datetime.datetime.now().strftime("%B %Y")
            periodo_actual = get_current_period()

            # Usar la pregunta expandida para buscar
            docs = _get_docs(expanded, category=force_category)

            if not docs:
                return "No tengo información sobre eso. Acude a Gestión Escolar (Edificio 1, Planta Baja) o llama al 57296000 ext. 52001."

            context_text = "\n---\n".join(d.page_content for d in docs[:3])

            history_text = ""
            if history_from_app:
                history_text = "\n".join([
                    f"Usuario: {h['user']}\nBot: {h['bot']}"
                    for h in history_from_app[-3:]
                ])

            # Ajustar instrucciones según categoría
            if force_category == "temario":
                topic_instruction = "Responde SOLO sobre bibliografía y contenido de temarios de unidades de aprendizaje de ESCOM."
                out_of_scope = "Solo puedo ayudarte con bibliografía y temarios de materias."
            else:
                topic_instruction = "Responde SOLO sobre becas, servicio social, estancia profesional e información institucional de ESCOM."
                out_of_scope = "Solo puedo ayudarte con becas, servicio social, estancia profesional y avisos institucionales."

            continuation_note = ""
            if is_continuation(question):
                continuation_note = "\nNOTA: El usuario pide MÁS información sobre el tema anterior. NO repitas lo que ya dijiste en el historial. Proporciona detalles adicionales, pasos siguientes o información complementaria.\n"

            prompt = f"""Eres ESCOMbot, asistente oficial de ESCOM IPN. {topic_instruction}
{continuation_note}
FECHA ACTUAL: {fecha_actual} | CICLO: {periodo_actual}

═══ REGLAS DE FORMATO (OBLIGATORIAS) ═══
1. ESTRUCTURA: Usa SIEMPRE este orden:
   - Primera línea: resumen en **negrita** (máx. 10 palabras)
   - Luego lista con viñetas (•), UNA por línea, con salto de línea entre cada una
   - Nunca escribas párrafos largos corridos

2. VIÑETAS: Cada punto en su propia línea así:
   • Punto uno
   • Punto dos
   • Punto tres

3. LÍMITES: Máximo 5 viñetas. Máximo 100 palabras en total.

4. FECHAS: Resáltalas con ⚠️ Ej: ⚠️ Fecha límite: 22 de mayo

5. FUERA DE CONTEXTO: Si no está en el CONTEXTO, responde exactamente:
   "No tengo esa información. Acude a Gestión Escolar (Edificio 1, PB) o llama al 57296000 ext. 52001."

6. VIGENCIA: Si el documento es de otro ciclo: "⚠️ Dato de periodo anterior — verifica vigencia:"

7. FUERA DE TEMA: "{out_of_scope}"

8. TONO: Directo. Sin "Claro que sí", sin "Por supuesto", sin introducción.

═══ CONTEXTO ═══
{context_text}

═══ HISTORIAL ═══
{history_text}

═══ PREGUNTA ═══
{question}

RESPUESTA (sigue el formato exacto, máximo 100 palabras):"""

            response = llm.invoke(prompt)
            answer = fix_incomplete_answer(response.content)
            # Solo cachear preguntas normales, no continuaciones
            if not is_continuation(question):
                _response_cache[cache_key] = answer
            return answer

        except Exception as e:
            import traceback
            print(f"[ERROR DETALLADO]\n{traceback.format_exc()}")
            print(f"[GROQ_API_KEY presente]: {bool(os.getenv('GROQ_API_KEY'))}")
            return "Lo siento, hubo un error técnico al procesar la pregunta."

    return chain