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

BASE_DIR = Path(__file__).resolve().parent.parent
FAISS_DIR = BASE_DIR / "data" / "faiss"
FAST_RESPONSES_PATH = BASE_DIR / "rag" / "fast_responses.json"
HORARIOS_PATH = BASE_DIR / "rag" / "horarios.json"
LINKS_PATH = BASE_DIR / "rag" / "links.json"

# ── Carga rápida solo de fast_responses (pequeño, siempre necesario) ──
try:
    with open(FAST_RESPONSES_PATH, "r", encoding="utf-8") as f:
        FAST_RESPONSES = json.load(f)
except FileNotFoundError:
    FAST_RESPONSES = []

# ── Horarios y links se cargan LAZY la primera vez que se usan ──
_HORARIOS: dict | None = None
_LINKS_DATA: dict | None = None
_LINKS_SECCIONES: dict | None = None

def _get_horarios() -> dict:
    global _HORARIOS
    if _HORARIOS is None:
        try:
            with open(HORARIOS_PATH, "r", encoding="utf-8") as f:
                _HORARIOS = json.load(f).get("profesores", {})
            print(f"[CHAIN] Horarios cargados lazy: {len(_HORARIOS)} profs")
        except FileNotFoundError:
            _HORARIOS = {}
    return _HORARIOS

def _get_links():
    global _LINKS_DATA, _LINKS_SECCIONES
    if _LINKS_DATA is None:
        try:
            with open(LINKS_PATH, "r", encoding="utf-8") as f:
                _LINKS_DATA = json.load(f)
            _LINKS_SECCIONES = {
                **_LINKS_DATA.get("institucionales", {}),
                **_LINKS_DATA.get("tramites", {}),
                **_LINKS_DATA.get("apoyo_estudiantil", {}),
            }
        except FileNotFoundError:
            _LINKS_DATA = {}
            _LINKS_SECCIONES = {}
    return _LINKS_DATA, _LINKS_SECCIONES

_response_cache: dict = {}
last_call_time = 0

# ─────────────────────────────────────────────
#  VALIDACIÓN
# ─────────────────────────────────────────────
def es_pregunta_valida(question: str) -> bool:
    q = question.strip()
    if len(q) < 3:
        return False
    if q.isdigit():
        return False
    letras = sum(1 for c in q if c.isalpha())
    if len(q) > 3 and letras / len(q) < 0.6:
        return False
    palabras = q.lower().split()
    if len(palabras) == 1:
        p = palabras[0]
        vocales = sum(1 for c in p if c in "aeiouáéíóú")
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
    horarios = _get_horarios()
    if not horarios:
        return None
    palabras_busqueda = [
        w.upper() for w in query.split()
        if len(w) > 2 and w.lower() not in STOP_WORDS_HORARIO
    ]
    if not palabras_busqueda:
        return None
    orden_dias = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"]
    resultados = []
    for nombre, clases in horarios.items():
        if any(word in nombre.upper() for word in palabras_busqueda):
            por_dia = {}
            for c in clases:
                por_dia.setdefault(c["dia"], []).append(
                    f"  ⏰ {c['entrada']}–{c['salida']} | {c['materia']} | 📍 Salón {c['salon']}"
                )
            lineas = [f"**{nombre.title()}**"]
            for dia in orden_dias:
                if dia in por_dia:
                    lineas.append(f"• {dia}:")
                    lineas.extend(por_dia[dia])
            resultados.append("\n".join(lineas))
    return "\n\n".join(resultados[:2]) if resultados else None

def es_pregunta_de_horario(query: str) -> bool:
    q = query.lower()
    return any(t in q for t in [
        "horario", "salón", "salon", "dónde da clase", "donde da clase",
        "cuándo da clase", "cuando da clase", "a qué hora", "a que hora",
        "qué día da", "que dia da", "dónde imparte", "donde imparte"
    ])

# ─────────────────────────────────────────────
#  LINKS
# ─────────────────────────────────────────────
def buscar_link(query: str) -> str | None:
    links_data, secciones = _get_links()
    if not links_data:
        return None
    q = query.lower()
    encontrados = []
    for key, palabras in links_data.get("keywords", {}).items():
        if any(p in q for p in palabras):
            link_data = secciones.get(key)
            if link_data:
                encontrados.append(f"• {link_data['nombre']}: {link_data['url']}")
    return ("**Links relevantes:**\n" + "\n".join(encontrados)) if encontrados else None

def es_pregunta_de_link(query: str) -> bool:
    q = query.lower()
    return any(t in q for t in [
        "link", "página", "pagina", "portal", "sitio", "web", "url",
        "dónde entro", "donde entro", "cómo accedo", "como accedo",
        "dónde me registro", "donde me registro"
    ])

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
    if now - last_call_time < 0.5:   # reducido de 1s a 0.5s
        return False
    last_call_time = now
    return True

def get_current_period():
    now = datetime.datetime.now()
    return f"{now.year}-{'2' if 2 <= now.month <= 7 else '1'}"

CONTINUATION_TRIGGERS = {
    "más", "mas", "más información", "más detalles", "continúa", "continua",
    "sigue", "y qué más", "que más", "dime más", "amplía", "amplia",
    "explica más", "detalla", "cuéntame más", "otro", "otra", "siguiente"
}

def is_continuation(q: str) -> bool:
    return q.lower().strip() in CONTINUATION_TRIGGERS

def expand_question(question: str, history: list) -> str:
    if not is_continuation(question) or not history:
        return question
    last = history[-1].get("user", "")
    return f"{last} — proporciona más detalles" if last else question

def classify_question(question: str) -> str:
    q = question.lower()
    if any(w in q for w in [
        "estancia profesional", "estancia", "siep", "horas estancia",
        "cuántas horas estancia", "cuantas horas estancia",
        "documentos estancia", "reporte estancia", "opción a", "opción b", "opción c",
        "qué pasa si reprueban", "extranjero estancia", "empresa estancia"
    ]):
        return "estancia_profesional"
    if any(w in q for w in [
        "servicio social", "servicio_social", "liberar servicio",
        "siss", "reporte mensual", "responsable directo", "480 horas",
        "liberación servicio", "baja servicio"
    ]):
        return "servicio_social"
    if any(w in q for w in ["beca", "becas", "sibec"]):
        return "becas"
    if any(w in q for w in [
        "temario", "bibliograf", "isbn", "editorial",
        "cálculo", "álgebra", "programación", "compiladores", "redes",
        "sistemas operativos", "bases de datos", "algoritmos",
        "automatas", "autómatas", "discretas"
    ]):
        return "temario"
    return ""

# ─────────────────────────────────────────────
#  LOAD CHAIN
# ─────────────────────────────────────────────
def load_chain():
    print(f"[CHAIN] BASE_DIR: {BASE_DIR} | FAISS existe: {FAISS_DIR.exists()}")

    from langchain_cohere import CohereEmbeddings
    embeddings = CohereEmbeddings(
        model="embed-multilingual-light-v3.0",
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )

    try:
        vectorstore = FAISS.load_local(
            str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True
        )
        print("[CHAIN] ✅ FAISS listo.")
    except Exception as e:
        print(f"[CHAIN] ❌ FAISS: {e}")
        return None

    GROQ_KEYS = [k.strip() for k in [
        os.getenv("GROQ_API_KEY", ""),
        os.getenv("GROQ_API_KEY_2", ""),
        os.getenv("GROQ_API_KEY_3", ""),
    ] if k.strip()]

    GROQ_MODELS = ["llama-3.1-8b-instant", "llama3-8b-8192", "gemma2-9b-it"]

    if not GROQ_KEYS:
        print("[CHAIN] ❌ Sin GROQ keys")
        return None

    _llm_options = [(key, model) for model in GROQ_MODELS for key in GROQ_KEYS]

    def _create_llm(key, model):
        return ChatGroq(model=model, temperature=0.0, max_tokens=600, api_key=key)

    def _get_docs(question: str, category: str = None):
        detected = category or classify_question(question)
        if detected:
            try:
                docs = vectorstore.as_retriever(
                    search_kwargs={"k": 5, "filter": {"category": detected}}
                ).invoke(question)
                if docs:
                    return docs
                print(f"[CHAIN] Sin docs con filtro={detected} — global")
            except Exception as e:
                print(f"[CHAIN] Filtro error: {e}")
        return vectorstore.as_retriever(search_kwargs={"k": 4}).invoke(question)

    def chain(question: str, history_from_app: list = None, force_category: str = None):
        if not can_call_model():
            return "Por favor espera un momento antes de enviar otra pregunta."

        # 1. Validar entrada
        if not es_pregunta_valida(question):
            return (
                "No entendí tu mensaje 😅\n"
                "Puedo ayudarte con:\n"
                "• Becas\n• Servicio Social\n• Estancia Profesional\n"
                "• Temarios\n• Horarios de profesores"
            )

        expanded = expand_question(question, history_from_app or [])
        cache_key = f"{force_category or ''}:{expanded.lower().strip()}"
        if not is_continuation(question) and cache_key in _response_cache:
            return _response_cache[cache_key]

        # 2. Respuestas rápidas (sin LLM)
        fast = fast_response(question)
        if fast:
            return fast

        # 3. Horario de profesor
        if es_pregunta_de_horario(question):
            horario = buscar_horario_profesor(question)
            if horario:
                return horario
            return (
                "No encontré ese profesor. Verifica el apellido.\n"
                "Ejemplo: *'horario de García'*\n"
                "🌐 https://www.escom.ipn.mx"
            )

        # 4. Links directos
        if es_pregunta_de_link(question):
            link_resp = buscar_link(question)
            if link_resp:
                return link_resp

        try:
            fecha_actual = datetime.datetime.now().strftime("%d de %B de %Y")
            docs = _get_docs(expanded, category=force_category)

            if not docs:
                link_resp = buscar_link(question)
                return (
                    f"No tengo información sobre eso.\n{link_resp}"
                    if link_resp else
                    "No tengo información. Acude a Gestión Escolar (Edif. 1, PB) o llama al 57296000 ext. 52001."
                )

            context_text = "\n---\n".join(d.page_content for d in docs[:4])

            history_text = ""
            if history_from_app:
                history_text = "\n".join([
                    f"Usuario: {h['user']}\nBot: {h['bot']}"
                    for h in history_from_app[-2:]
                ])

            detected_cat = force_category or classify_question(question)

            instrucciones = {
                "estancia_profesional": (
                    "Responde sobre ESTANCIA PROFESIONAL de ESCOM. "
                    "Es una materia curricular (no servicio social): empresa privada/gobierno TI, "
                    "máximo 200 horas, sistema SIEP. NO confundas con servicio social.",
                    "Solo respondo sobre estancia profesional de ESCOM."
                ),
                "servicio_social": (
                    "Responde sobre SERVICIO SOCIAL de ESCOM. "
                    "Mínimo 480 horas en 6 meses, sistema SISS, gestiona la UPIS.",
                    "Solo respondo sobre servicio social de ESCOM."
                ),
                "temario": (
                    "Responde sobre temarios y bibliografía de materias de ESCOM IPN. "
                    "Incluye temas principales y bibliografía (autor, año, título, ISBN).",
                    "Solo respondo sobre temarios de ESCOM."
                ),
                "becas": (
                    "Responde sobre becas del IPN y ESCOM.",
                    "Solo respondo sobre becas de IPN/ESCOM."
                ),
            }
            topic_instruction, out_of_scope = instrucciones.get(
                detected_cat,
                ("Responde sobre becas, servicio social, estancia e información de ESCOM.", "Solo respondo sobre trámites de ESCOM.")
            )

            continuation_note = "\nNOTA: Da MÁS detalles sin repetir lo anterior.\n" if is_continuation(question) else ""

            prompt = f"""Eres ESCOMbot, asistente de ESCOM IPN. {topic_instruction}
{continuation_note}FECHA: {fecha_actual} | CICLO: {get_current_period()}

REGLAS:
1. Primera línea en **negrita** (resumen, máx 10 palabras)
2. Viñetas (•), una por línea, máx 6 viñetas, máx 150 palabras
3. Fechas con ⚠️ Ej: ⚠️ Límite: 22 mayo 2026
4. Links con 🔗 Ej: 🔗 https://siep.escom.ipn.mx
5. USA el CONTEXTO aunque la pregunta esté redactada diferente
6. Si no hay datos en el contexto: "No tengo esa información. Acude a Gestión Escolar."
7. Fuera de tema: "{out_of_scope}"
8. Sin introducciones ni "Claro que sí"

CONTEXTO:
{context_text}

HISTORIAL:
{history_text}

PREGUNTA: {question}

RESPUESTA:"""

            response = None
            for key, model in _llm_options:
                try:
                    response = _create_llm(key, model).invoke(prompt)
                    break
                except Exception as e:
                    if "429" in str(e) or "rate_limit" in str(e).lower():
                        continue
                    raise

            if response is None:
                return "El servicio alcanzó su límite. Intenta en unos minutos."

            answer = fix_incomplete_answer(response.content)

            # Agregar link automático si no lo incluyó el modelo
            if detected_cat == "estancia_profesional" and "siep.escom.ipn.mx" not in answer:
                answer += "\n🔗 SIEP: https://siep.escom.ipn.mx"
            elif detected_cat == "becas" and "sibec.ipn.mx" not in answer:
                answer += "\n🔗 SIBec: https://www.sibec.ipn.mx"

            if not is_continuation(question):
                _response_cache[cache_key] = answer
            return answer

        except Exception:
            import traceback
            print(f"[ERROR]\n{traceback.format_exc()}")
            return "Error técnico al procesar la pregunta."

    return chain