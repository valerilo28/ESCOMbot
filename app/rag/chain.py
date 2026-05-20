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
HORARIOS_PATH = BASE_DIR / "rag" / "horarios.json"

try:
    with open(FAST_RESPONSES_PATH, "r", encoding="utf-8") as f:
        FAST_RESPONSES = json.load(f)
except FileNotFoundError:
    print(f"[WARN] fast_responses.json no encontrado")
    FAST_RESPONSES = []

try:
    with open(HORARIOS_PATH, "r", encoding="utf-8") as f:
        HORARIOS = json.load(f).get("profesores", {})
except FileNotFoundError:
    HORARIOS = {}

def buscar_horario_profesor(query: str) -> str:
    """Busca el horario de un profesor por nombre parcial."""
    if not HORARIOS:
        return None
    query_upper = query.upper()
    palabras = [p for p in query_upper.split() if len(p) > 3]
    resultados = []
    for nombre, clases in HORARIOS.items():
        if any(palabra in nombre for palabra in palabras):
            dias_str = []
            for c in clases:
                dias_str.append(f"• {c['dia']} {c['entrada']}-{c['salida']} — {c['materia']} (Salón {c['salon']})")
            resultados.append(f"**{nombre.title()}:**\n" + "\n".join(dias_str))
    if resultados:
        return "\n\n".join(resultados[:2])
    return None

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

def classify_question(question: str) -> str:
    q = question.lower()
    if "beca" in q:
        return "becas"
    elif "servicio social" in q or "servicio_social" in q:
        return "servicio_social"
    elif "estancia" in q:
        return "estancia_profesional"
    elif any(w in q for w in [
        "temario", "bibliograf", "materia", "unidad de aprendizaje",
        "libro", "autor", "isbn", "editorial", "temas de", "contenido de",
        "teoría de", "cálculo", "álgebra", "programación", "compiladores",
        "redes", "sistemas operativos", "bases de datos", "algoritmos",
        "automatas", "autómatas", "computación", "discretas"
    ]):
        return "temario"
    elif any(w in q for w in [
        "historia", "misión", "visión", "carrera", "fundada", "cosecovi",
        "academia", "ubicación", "dirección", "correo escom"
    ]):
        return "general"
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

    # 3. Configurar LLM con fallback de keys y modelos
    # Orden: intenta cada key con el modelo principal, luego baja de modelo
    GROQ_KEYS = [
        os.getenv("GROQ_API_KEY", "").strip(),
        os.getenv("GROQ_API_KEY_2", "").strip(),
        os.getenv("GROQ_API_KEY_3", "").strip(),
    ]
    GROQ_MODELS = [
        "llama-3.1-8b-instant",   # más rápido
        "llama3-8b-8192",          # fallback 1
        "gemma2-9b-it",            # fallback 2
    ]
    GROQ_KEYS = [k for k in GROQ_KEYS if k]  # quitar vacías

    print(f"[CHAIN] GROQ keys disponibles: {len(GROQ_KEYS)}")
    print(f"[CHAIN] GROQ modelos en orden: {GROQ_MODELS}")

    if not GROQ_KEYS:
        print("[CHAIN] ❌ No hay GROQ_API_KEY configurada")
        return None

    # Crear lista de (key, modelo) en orden de prioridad
    # Primero todas las keys con el modelo 1, luego con modelo 2, etc.
    _llm_options = [
        (key, model)
        for model in GROQ_MODELS
        for key in GROQ_KEYS
    ]

    def _create_llm(key: str, model: str):
        return ChatGroq(
            model=model,
            temperature=0.0,
            max_tokens=700,
            api_key=key
        )

    # LLM inicial con la primera key y modelo
    llm = _create_llm(GROQ_KEYS[0], GROQ_MODELS[0])
    print(f"[CHAIN] ✅ LLM Groq listo: {GROQ_MODELS[0]}")

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

        # Buscar horario de profesor si la pregunta lo pide
        q_lower = question.lower()
        if any(w in q_lower for w in ["horario", "clase", "salón", "salon", "dónde da clase", "cuándo da clase", "horario del profesor", "horario de"]):
            horario = buscar_horario_profesor(question)
            if horario:
                return f"**Horario encontrado:**\n{horario}"

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
                topic_instruction = """Responde sobre temarios y bibliografía de materias de ESCOM IPN.
Cuando pregunten por una materia, proporciona:
1. Los temas principales que se estudian
2. La bibliografía básica recomendada (autor, año, título, editorial/ISBN)
3. Recursos digitales si los hay
NO menciones que es el temario de una carrera específica, solo di que es el programa de estudios de ESCOM."""
                out_of_scope = "Solo puedo ayudarte con temarios y bibliografía de materias de ESCOM."
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

            # E. Ejecutar con fallback de keys y modelos
            response = None
            last_error = None
            for key, model in _llm_options:
                try:
                    current_llm = _create_llm(key, model)
                    response = current_llm.invoke(prompt)
                    print(f"[CHAIN] Respondió con modelo={model}")
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "rate_limit" in err_str.lower():
                        print(f"[CHAIN] Rate limit key={key[:20]}... modelo={model}, probando siguiente...")
                        last_error = e
                        continue
                    else:
                        raise

            if response is None:
                return "El servicio de IA alcanzó su límite temporalmente. Intenta de nuevo en unos minutos."

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