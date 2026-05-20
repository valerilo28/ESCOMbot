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
    if not HORARIOS:
        return None
    query_upper = query.upper()
    palabras = [p for p in query_upper.split() if len(p) > 3]
    resultados = []
    for nombre, clases in HORARIOS.items():
        if any(palabra in nombre for palabra in palabras):
            dias_str = [
                f"• {c['dia']} {c['entrada']}-{c['salida']} — {c['materia']} (Salón {c['salon']})"
                for c in clases
            ]
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
    answer = answer.strip()
    answer = re.sub(r'```[\w]*\n?', '', answer).strip()
    answer = re.sub(r'^[\-\*]\s+', '• ', answer, flags=re.MULTILINE)
    answer = re.sub(r'(?<!\n)(• )', r'\n\1', answer)
    answer = re.sub(r'\n{3,}', '\n\n', answer)
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
    if not is_continuation(question) or not history:
        return question
    last_user_q = history[-1].get("user", "") if history else ""
    if last_user_q:
        return f"{last_user_q} — proporciona más detalles o información adicional"
    return question

def classify_question(question: str) -> str:
    """Clasifica la pregunta en una categoría para filtrar el vectorstore."""
    q = question.lower()

    # Servicio social — ampliado con más variantes
    if any(w in q for w in [
        "servicio social", "servicio_social", "ss ", " ss,", "liberar servicio",
        "liberación servicio", "baja servicio", "art 91", "artículo 91",
        "carta de presentación servicio", "dictamen electiva", "dictamen menos 70"
    ]):
        return "servicio_social"

    # Estancia profesional — ampliado
    elif any(w in q for w in [
        "estancia", "estancia profesional", "acreditación estancia",
        "requisitos estancia", "dictamen estancia", "empresa estancia",
        "carta estancia", "reporte estancia"
    ]):
        return "estancia_profesional"

    # Becas
    elif any(w in q for w in ["beca", "becas", "convocatoria beca", "apoyo económico"]):
        return "becas"

    # Temarios
    elif any(w in q for w in [
        "temario", "bibliograf", "materia", "unidad de aprendizaje",
        "libro", "autor", "isbn", "editorial", "temas de", "contenido de",
        "teoría de", "cálculo", "álgebra", "programación", "compiladores",
        "redes", "sistemas operativos", "bases de datos", "algoritmos",
        "automatas", "autómatas", "computación", "discretas"
    ]):
        return "temario"

    # General institucional
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

    from langchain_cohere import CohereEmbeddings
    cohere_key = os.getenv("COHERE_API_KEY")
    print(f"[CHAIN] COHERE_API_KEY presente: {bool(cohere_key)}")
    embeddings = CohereEmbeddings(
        model="embed-multilingual-light-v3.0",
        cohere_api_key=cohere_key
    )
    print("[CHAIN] Embeddings listos.")

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

    GROQ_KEYS = [
        os.getenv("GROQ_API_KEY", "").strip(),
        os.getenv("GROQ_API_KEY_2", "").strip(),
        os.getenv("GROQ_API_KEY_3", "").strip(),
    ]
    GROQ_MODELS = [
        "llama-3.1-8b-instant",
        "llama3-8b-8192",
        "gemma2-9b-it",
    ]
    GROQ_KEYS = [k for k in GROQ_KEYS if k]

    print(f"[CHAIN] GROQ keys disponibles: {len(GROQ_KEYS)}")

    if not GROQ_KEYS:
        print("[CHAIN] ❌ No hay GROQ_API_KEY configurada")
        return None

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

    llm = _create_llm(GROQ_KEYS[0], GROQ_MODELS[0])
    print(f"[CHAIN] ✅ LLM Groq listo: {GROQ_MODELS[0]}")

    def _get_docs(question: str, category: str = None):
        """Recupera documentos. Si hay categoría, filtra por ella.
        Si el filtro no retorna resultados, hace búsqueda sin filtro como fallback.
        """
        detected = category or classify_question(question)

        if detected:
            # Intento 1: con filtro por categoría
            try:
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 5, "filter": {"category": detected}}
                )
                docs = retriever.invoke(question)
                if docs:
                    print(f"[CHAIN] ✅ {len(docs)} docs encontrados con filtro category={detected}")
                    return docs
                else:
                    print(f"[CHAIN] ⚠️ Filtro category={detected} sin resultados — usando búsqueda global")
            except Exception as e:
                print(f"[CHAIN] ⚠️ Error con filtro: {e} — usando búsqueda global")

        # Fallback: búsqueda sin filtro (más amplia)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(question)
        print(f"[CHAIN] Búsqueda global: {len(docs)} docs")
        return docs

    def chain(question: str, history_from_app: list = None, force_category: str = None):
        if not can_call_model():
            return "Por favor, espera un momento antes de enviar otra pregunta."

        expanded = expand_question(question, history_from_app or [])

        cache_key = f"{force_category or ''}:{expanded.lower().strip()}"
        if not is_continuation(question) and cache_key in _response_cache:
            return _response_cache[cache_key]

        fast = fast_response(question)
        if fast:
            return fast

        q_lower = question.lower()
        if any(w in q_lower for w in ["horario", "clase", "salón", "salon", "dónde da clase", "cuándo da clase"]):
            horario = buscar_horario_profesor(question)
            if horario:
                return f"**Horario encontrado:**\n{horario}"

        try:
            fecha_actual = datetime.datetime.now().strftime("%B %Y")
            periodo_actual = get_current_period()

            docs = _get_docs(expanded, category=force_category)

            if not docs:
                return "No tengo información sobre eso en este momento. Acude a Gestión Escolar (Edificio 1, Planta Baja) o llama al 57296000 ext. 52001."

            context_text = "\n---\n".join(d.page_content for d in docs[:4])

            # Incluir de qué archivos vienen los docs (útil para depurar)
            sources = list({d.metadata.get("filename", "?") for d in docs})
            print(f"[CHAIN] Fuentes usadas: {sources}")

            history_text = ""
            if history_from_app:
                history_text = "\n".join([
                    f"Usuario: {h['user']}\nBot: {h['bot']}"
                    for h in history_from_app[-3:]
                ])

            # Detectar categoría para ajustar instrucciones
            detected_cat = force_category or classify_question(question)

            if detected_cat == "temario":
                topic_instruction = """Responde sobre temarios y bibliografía de materias de ESCOM IPN.
Cuando pregunten por una materia, proporciona:
1. Los temas principales que se estudian
2. La bibliografía básica recomendada (autor, año, título, editorial/ISBN)
3. Recursos digitales si los hay
NO menciones que es el temario de una carrera específica."""
                out_of_scope = "Solo puedo ayudarte con temarios y bibliografía de materias de ESCOM."
            elif detected_cat == "servicio_social":
                topic_instruction = "Responde SOLO sobre servicio social de ESCOM IPN. Usa el contexto proporcionado."
                out_of_scope = "Solo puedo ayudarte con temas de servicio social, becas, estancia y avisos de ESCOM."
            elif detected_cat == "estancia_profesional":
                topic_instruction = "Responde SOLO sobre estancia profesional de ESCOM IPN. Usa el contexto proporcionado."
                out_of_scope = "Solo puedo ayudarte con temas de estancia, servicio social, becas y avisos de ESCOM."
            else:
                topic_instruction = "Responde SOLO sobre becas, servicio social, estancia profesional e información institucional de ESCOM."
                out_of_scope = "Solo puedo ayudarte con becas, servicio social, estancia profesional y avisos institucionales."

            continuation_note = ""
            if is_continuation(question):
                continuation_note = "\nNOTA: El usuario pide MÁS información sobre el tema anterior. NO repitas lo que ya dijiste. Proporciona detalles adicionales o pasos siguientes.\n"

            prompt = f"""Eres ESCOMbot, asistente oficial de ESCOM IPN. {topic_instruction}
{continuation_note}
FECHA ACTUAL: {fecha_actual} | CICLO: {periodo_actual}

═══ REGLAS DE FORMATO (OBLIGATORIAS) ═══
1. ESTRUCTURA:
   - Primera línea: resumen en **negrita** (máx. 10 palabras)
   - Lista con viñetas (•), UNA por línea
   - Sin párrafos largos

2. VIÑETAS — cada punto en su propia línea:
   • Punto uno
   • Punto dos

3. LÍMITES: Máximo 5 viñetas. Máximo 120 palabras en total.

4. FECHAS: Resáltalas con ⚠️ Ej: ⚠️ Fecha límite: 22 de mayo

5. USA EL CONTEXTO: Si la respuesta está en el CONTEXTO de abajo, ÚSALA aunque la pregunta esté formulada de forma diferente.
   Solo responde "No tengo esa información" si el contexto realmente no contiene datos relevantes.

6. VIGENCIA: Si el documento es de otro ciclo: "⚠️ Dato de periodo anterior — verifica vigencia:"

7. FUERA DE TEMA: "{out_of_scope}"

8. TONO: Directo. Sin "Claro que sí", sin "Por supuesto", sin introducción.

═══ CONTEXTO (documentos recuperados) ═══
{context_text}

═══ HISTORIAL ═══
{history_text}

═══ PREGUNTA ═══
{question}

RESPUESTA (sigue el formato exacto, máximo 120 palabras):"""

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
                        print(f"[CHAIN] Rate limit — probando siguiente...")
                        last_error = e
                        continue
                    else:
                        raise

            if response is None:
                return "El servicio de IA alcanzó su límite temporalmente. Intenta de nuevo en unos minutos."

            answer = fix_incomplete_answer(response.content)
            if not is_continuation(question):
                _response_cache[cache_key] = answer
            return answer

        except Exception as e:
            import traceback
            print(f"[ERROR DETALLADO]\n{traceback.format_exc()}")
            return "Lo siento, hubo un error técnico al procesar la pregunta."

    return chain