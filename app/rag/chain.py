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
LINKS_PATH = BASE_DIR / "rag" / "links.json"

# --- CARGAR DATOS ---
try:
    with open(FAST_RESPONSES_PATH, "r", encoding="utf-8") as f:
        FAST_RESPONSES = json.load(f)
except FileNotFoundError:
    print("[WARN] fast_responses.json no encontrado")
    FAST_RESPONSES = []

try:
    with open(HORARIOS_PATH, "r", encoding="utf-8") as f:
        HORARIOS = json.load(f).get("profesores", {})
    print(f"[CHAIN] Horarios cargados: {len(HORARIOS)} profesores")
except FileNotFoundError:
    HORARIOS = {}
    print("[WARN] horarios.json no encontrado")

try:
    with open(LINKS_PATH, "r", encoding="utf-8") as f:
        LINKS_DATA = json.load(f)
except FileNotFoundError:
    LINKS_DATA = {}
    print("[WARN] links.json no encontrado")

_response_cache: dict = {}
last_call_time = 0

# ─────────────────────────────────────────────
#  VALIDACIÓN — detectar si la pregunta es basura
# ─────────────────────────────────────────────
def es_pregunta_valida(question: str) -> bool:
    """Rechaza entradas que no son preguntas reales: texto aleatorio, muy corto sin sentido, etc."""
    q = question.strip()
    # Muy corta (< 3 chars)
    if len(q) < 3:
        return False
    # Solo números
    if q.isdigit():
        return False
    # Mayoría de caracteres no son letras ni espacios (texto aleatorio tipo "dlkdo", "asdfgh")
    letras = sum(1 for c in q if c.isalpha())
    if len(q) > 3 and letras / len(q) < 0.6:
        return False
    # Palabras sin sentido: si tiene solo 1 "palabra" de 3-8 chars sin vocales
    palabras = q.lower().split()
    if len(palabras) == 1:
        p = palabras[0]
        vocales = sum(1 for c in p if c in "aeiouáéíóú")
        if len(p) >= 4 and vocales == 0:
            return False
        # Palabras tipo "dlkdo", "asdfg" — demasiadas consonantes seguidas
        if len(p) >= 4 and vocales / len(p) < 0.2:
            return False
    return True


# ─────────────────────────────────────────────
#  HORARIOS
# ─────────────────────────────────────────────
STOP_WORDS_HORARIO = {
    "horario", "del", "de", "la", "el", "los", "las", "profesor", "profesora",
    "maestra", "maestro", "profe", "donde", "dónde", "salón", "salon",
    "clase", "clases", "da", "tiene", "cuando", "cuándo", "en", "qué", "que",
    "está", "esta", "cuál", "cual", "dar", "imparte", "dicta"
}

def buscar_horario_profesor(query: str) -> str | None:
    if not HORARIOS:
        return None

    palabras_busqueda = [
        w.upper() for w in query.split()
        if len(w) > 2 and w.lower() not in STOP_WORDS_HORARIO
    ]
    if not palabras_busqueda:
        return None

    resultados = []
    for nombre, clases in HORARIOS.items():
        nombre_upper = nombre.upper()
        if any(word in nombre_upper for word in palabras_busqueda):
            orden_dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"]
            por_dia = {}
            for c in clases:
                dia = c["dia"]
                if dia not in por_dia:
                    por_dia[dia] = []
                por_dia[dia].append(
                    f"  ⏰ {c['entrada']}–{c['salida']} | {c['materia']} | 📍 Salón {c['salon']}"
                )
            lineas = [f"**{nombre.title()}**"]
            for dia in orden_dias:
                if dia in por_dia:
                    lineas.append(f"• {dia}:")
                    lineas.extend(por_dia[dia])
            resultados.append("\n".join(lineas))

    if resultados:
        return "\n\n".join(resultados[:2])
    return None


def es_pregunta_de_horario(query: str) -> bool:
    q = query.lower()
    triggers = [
        "horario", "salón", "salon", "dónde da clase", "donde da clase",
        "cuándo da clase", "cuando da clase", "horario del profesor",
        "horario de ", "a qué hora", "a que hora", "qué día da",
        "que dia da", "dónde está el profesor", "donde esta el profesor",
        "ubicación del profesor", "dónde imparte", "donde imparte"
    ]
    return any(t in q for t in triggers)


# ─────────────────────────────────────────────
#  LINKS
# ─────────────────────────────────────────────
def buscar_link(query: str) -> str | None:
    if not LINKS_DATA:
        return None
    q = query.lower()
    keywords_map = LINKS_DATA.get("keywords", {})
    secciones = {
        **LINKS_DATA.get("institucionales", {}),
        **LINKS_DATA.get("tramites", {}),
        **LINKS_DATA.get("apoyo_estudiantil", {}),
    }
    encontrados = []
    for key, palabras in keywords_map.items():
        if any(p in q for p in palabras):
            link_data = secciones.get(key)
            if link_data:
                encontrados.append(f"• {link_data['nombre']}: {link_data['url']}")
    if encontrados:
        return "**Links relevantes:**\n" + "\n".join(encontrados)
    return None


def es_pregunta_de_link(query: str) -> bool:
    q = query.lower()
    triggers = [
        "link", "página", "pagina", "portal", "sitio", "web", "url",
        "dirección web", "dónde entro", "donde entro", "cómo accedo",
        "como accedo", "dónde me registro", "donde me registro"
    ]
    return any(t in q for t in triggers)


# ─────────────────────────────────────────────
#  RESPUESTAS RÁPIDAS
# ─────────────────────────────────────────────
def fast_response(question: str):
    q = question.lower()
    for item in FAST_RESPONSES:
        for keyword in item["keywords"]:
            if keyword in q:
                return item["answer"]
    return None


# ─────────────────────────────────────────────
#  UTILIDADES
# ─────────────────────────────────────────────
def fix_incomplete_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r'```[\w]*\n?', '', answer).strip()
    answer = re.sub(r'^[\-\*]\s+', '• ', answer, flags=re.MULTILINE)
    answer = re.sub(r'(?<!\n)(• )', r'\n\1', answer)
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    if not answer.endswith((".", "!", "?", ":", ")")):
        sentences = re.split(r'(?<=[.!?]) +', answer)
        answer = " ".join(sentences[:-1]) if len(sentences) > 1 else answer + "..."
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
    return f"{last_user_q} — proporciona más detalles o información adicional" if last_user_q else question


def classify_question(question: str) -> str:
    """Clasifica con prioridad correcta: estancia ANTES que servicio social."""
    q = question.lower()

    # Estancia profesional — primero, porque "estancia" no debe clasificarse como servicio social
    if any(w in q for w in [
        "estancia profesional", "estancia", "acreditación estancia",
        "requisitos estancia", "dictamen estancia", "empresa estancia",
        "carta estancia", "reporte estancia", "siep", "horas estancia",
        "cuántas horas estancia", "cuantas horas estancia",
        "documentos estancia", "qué pasa si reprueban reporte",
        "extranjero estancia", "opción a", "opción b", "opción c"
    ]):
        return "estancia_profesional"

    # Servicio social
    elif any(w in q for w in [
        "servicio social", "servicio_social", "liberar servicio",
        "liberación servicio", "baja servicio", "art 91", "artículo 91",
        "carta de presentación servicio", "dictamen electiva", "siss",
        "reporte mensual", "responsable directo", "480 horas"
    ]):
        return "servicio_social"

    # Becas
    elif any(w in q for w in ["beca", "becas", "sibec", "convocatoria beca", "apoyo económico"]):
        return "becas"

    # Temarios
    elif any(w in q for w in [
        "temario", "bibliograf", "materia", "unidad de aprendizaje",
        "libro", "autor", "isbn", "editorial", "temas de", "contenido de",
        "cálculo", "álgebra", "programación", "compiladores", "redes",
        "sistemas operativos", "bases de datos", "algoritmos",
        "automatas", "autómatas", "computación", "discretas"
    ]):
        return "temario"

    elif any(w in q for w in [
        "historia", "misión", "visión", "carrera", "fundada", "cosecovi",
        "academia", "ubicación escom", "dirección escom"
    ]):
        return "general"

    return ""


# ─────────────────────────────────────────────
#  LOAD CHAIN
# ─────────────────────────────────────────────
def load_chain():
    print(f"[CHAIN] BASE_DIR: {BASE_DIR}")
    print(f"[CHAIN] FAISS: {FAISS_DIR} — existe: {FAISS_DIR.exists()}")

    from langchain_cohere import CohereEmbeddings
    cohere_key = os.getenv("COHERE_API_KEY")
    embeddings = CohereEmbeddings(
        model="embed-multilingual-light-v3.0",
        cohere_api_key=cohere_key
    )

    try:
        vectorstore = FAISS.load_local(
            str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True
        )
        print("[CHAIN] ✅ FAISS cargado.")
    except Exception as e:
        print(f"[CHAIN] ❌ Error FAISS: {e}")
        return None

    GROQ_KEYS = [k.strip() for k in [
        os.getenv("GROQ_API_KEY", ""),
        os.getenv("GROQ_API_KEY_2", ""),
        os.getenv("GROQ_API_KEY_3", ""),
    ] if k.strip()]

    GROQ_MODELS = ["llama-3.1-8b-instant", "llama3-8b-8192", "gemma2-9b-it"]

    if not GROQ_KEYS:
        print("[CHAIN] ❌ Sin GROQ_API_KEY")
        return None

    _llm_options = [(key, model) for model in GROQ_MODELS for key in GROQ_KEYS]

    def _create_llm(key, model):
        return ChatGroq(model=model, temperature=0.0, max_tokens=700, api_key=key)

    print(f"[CHAIN] ✅ LLM listo. Keys: {len(GROQ_KEYS)}")

    def _get_docs(question: str, category: str = None):
        """Busca con filtro de categoría; si no encuentra, hace búsqueda global."""
        detected = category or classify_question(question)

        if detected:
            try:
                docs = vectorstore.as_retriever(
                    search_kwargs={"k": 6, "filter": {"category": detected}}
                ).invoke(question)
                if docs:
                    print(f"[CHAIN] ✅ {len(docs)} docs — category={detected}")
                    return docs
                print(f"[CHAIN] ⚠️ Sin docs con filtro={detected} — búsqueda global")
            except Exception as e:
                print(f"[CHAIN] ⚠️ Error filtro: {e}")

        docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(question)
        print(f"[CHAIN] Búsqueda global: {len(docs)} docs")
        return docs

    # ── CHAIN PRINCIPAL ──
    def chain(question: str, history_from_app: list = None, force_category: str = None):
        if not can_call_model():
            return "Por favor, espera un momento antes de enviar otra pregunta."

        # ── 1. VALIDAR que la pregunta tiene sentido ──
        if not es_pregunta_valida(question):
            return (
                "No entendí tu mensaje. 😅\n"
                "Puedo ayudarte con:\n"
                "• Becas\n"
                "• Servicio Social\n"
                "• Estancia Profesional\n"
                "• Temarios y bibliografía\n"
                "• Horarios de profesores\n"
                "¿En qué te puedo ayudar?"
            )

        expanded = expand_question(question, history_from_app or [])
        cache_key = f"{force_category or ''}:{expanded.lower().strip()}"

        if not is_continuation(question) and cache_key in _response_cache:
            return _response_cache[cache_key]

        # ── 2. RESPUESTAS RÁPIDAS ──
        fast = fast_response(question)
        if fast:
            return fast

        # ── 3. HORARIO DE PROFESOR ──
        if es_pregunta_de_horario(question):
            horario = buscar_horario_profesor(question)
            if horario:
                return horario
            # Intentar con palabras de la pregunta directamente
            return (
                "No encontré ese profesor. Verifica el apellido o nombre completo.\n"
                "Ejemplo: *'horario de García Aguilar'*\n"
                "🌐 También puedes consultar: https://www.escom.ipn.mx"
            )

        # ── 4. LINKS DIRECTOS ──
        if es_pregunta_de_link(question):
            link_resp = buscar_link(question)
            if link_resp:
                return link_resp

        try:
            fecha_actual = datetime.datetime.now().strftime("%d de %B de %Y")
            periodo_actual = get_current_period()

            docs = _get_docs(expanded, category=force_category)

            if not docs:
                link_resp = buscar_link(question)
                if link_resp:
                    return f"No encontré información detallada, pero aquí tienes:\n{link_resp}"
                return (
                    "No tengo información sobre eso.\n"
                    "Acude a Gestión Escolar (Edificio 1, Planta Baja) "
                    "o llama al 57296000 ext. 52001."
                )

            context_text = "\n---\n".join(d.page_content for d in docs[:5])
            sources = list({d.metadata.get("filename", "?") for d in docs})
            print(f"[CHAIN] Fuentes: {sources}")

            history_text = ""
            if history_from_app:
                history_text = "\n".join([
                    f"Usuario: {h['user']}\nBot: {h['bot']}"
                    for h in history_from_app[-3:]
                ])

            detected_cat = force_category or classify_question(question)

            if detected_cat == "estancia_profesional":
                topic_instruction = (
                    "Responde SOLO sobre estancia profesional de ESCOM IPN. "
                    "La estancia profesional es diferente al servicio social: "
                    "es una materia curricular, se realiza en empresa privada o gobierno, "
                    "requiere 200 horas y el sistema es el SIEP (https://siep.escom.ipn.mx). "
                    "NO confundas con servicio social."
                )
                out_of_scope = "Solo puedo ayudarte con temas de estancia profesional de ESCOM."
            elif detected_cat == "servicio_social":
                topic_instruction = (
                    "Responde SOLO sobre servicio social de ESCOM IPN. "
                    "El servicio social requiere mínimo 480 horas en 6 meses, "
                    "el sistema es el SISS, y gestiona la UPIS."
                )
                out_of_scope = "Solo puedo ayudarte con temas de servicio social de ESCOM."
            elif detected_cat == "temario":
                topic_instruction = (
                    "Responde sobre temarios y bibliografía de materias de ESCOM IPN. "
                    "Proporciona temas principales y bibliografía recomendada (autor, año, título, ISBN)."
                )
                out_of_scope = "Solo puedo ayudarte con temarios y bibliografía de ESCOM."
            elif detected_cat == "becas":
                topic_instruction = "Responde sobre becas del IPN y ESCOM usando el contexto."
                out_of_scope = "Solo puedo ayudarte con becas de IPN/ESCOM."
            else:
                topic_instruction = "Responde sobre becas, servicio social, estancia profesional e información institucional de ESCOM."
                out_of_scope = "Solo puedo ayudarte con trámites e información de ESCOM."

            continuation_note = ""
            if is_continuation(question):
                continuation_note = "\nNOTA: El usuario pide MÁS información. NO repitas lo ya dicho. Da detalles adicionales.\n"

            prompt = f"""Eres ESCOMbot, asistente oficial de ESCOM IPN.
{topic_instruction}
{continuation_note}
FECHA ACTUAL: {fecha_actual} | CICLO: {periodo_actual}

═══ REGLAS OBLIGATORIAS ═══
1. FORMATO:
   - Primera línea: resumen en **negrita** (máx. 10 palabras)
   - Lista de viñetas (•), UNA por línea con salto de línea entre cada una
   - Sin párrafos largos corridos

2. VIÑETAS — cada punto en su línea:
   • Punto uno
   • Punto dos

3. LÍMITES: Máximo 6 viñetas. Máximo 150 palabras.

4. FECHAS LÍMITE: Si el contexto tiene fechas, SIEMPRE inclúyelas con ⚠️
   Ejemplo: ⚠️ Fecha límite: 22 de mayo de 2026

5. LINKS: Si el trámite tiene sistema en línea, incluye el link al final con 🔗
   Ejemplo: 🔗 Sistema: https://siep.escom.ipn.mx

6. USA EL CONTEXTO: Si la respuesta está en el CONTEXTO, ÚSALA aunque la pregunta esté redactada diferente.
   SOLO di "No tengo esa información" si el contexto genuinamente no tiene datos relevantes.

7. VIGENCIA: Si el documento es de otro ciclo escolar: ⚠️ Dato de periodo anterior — verifica vigencia.

8. FUERA DE TEMA: "{out_of_scope}"

9. TONO: Directo. Sin "Claro que sí", sin "Por supuesto", sin introducción.

═══ CONTEXTO (documentos recuperados de los PDFs) ═══
{context_text}

═══ HISTORIAL ═══
{history_text}

═══ PREGUNTA DEL USUARIO ═══
{question}

RESPUESTA (máximo 150 palabras, viñetas, incluye fechas y links si aplica):"""

            response = None
            for key, model in _llm_options:
                try:
                    response = _create_llm(key, model).invoke(prompt)
                    print(f"[CHAIN] ✅ Modelo: {model}")
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "rate_limit" in err_str.lower() or "RESOURCE_EXHAUSTED" in err_str:
                        print(f"[CHAIN] Rate limit — siguiente...")
                        continue
                    raise

            if response is None:
                return "El servicio de IA alcanzó su límite. Intenta en unos minutos."

            answer = fix_incomplete_answer(response.content)

            # Agregar link al final si la respuesta no lo incluyó y aplica
            if detected_cat == "estancia_profesional" and "siep.escom.ipn.mx" not in answer:
                answer += "\n🔗 Sistema SIEP: https://siep.escom.ipn.mx"
            elif detected_cat == "becas" and "sibec.ipn.mx" not in answer:
                answer += "\n🔗 SIBec: https://www.sibec.ipn.mx"

            if not is_continuation(question):
                _response_cache[cache_key] = answer
            return answer

        except Exception as e:
            import traceback
            print(f"[ERROR]\n{traceback.format_exc()}")
            return "Lo siento, hubo un error técnico al procesar la pregunta."

    return chain