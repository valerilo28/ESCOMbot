import os
import time
import datetime
from pathlib import Path
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
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

with open(FAST_RESPONSES_PATH, "r", encoding="utf-8") as f:
    FAST_RESPONSES = json.load(f)

# --- LÓGICA DE CONTROL ---

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
        # intenta cortar hasta la última oración completa
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
    # Semestre 2: Febrero a Julio, Semestre 1: Agosto a Enero
    semester = "2" if 2 <= now.month <= 7 else "1"
    return f"{now.year}-{semester}"

    cache = {}

def load_chain():
    # 1. Configurar Embeddings (Deben ser los mismos que usaste al crear el índice)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 2. Cargar base de datos vectorial FAISS
    try:
        vectorstore = FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("Base de datos FAISS cargada correctamente.")
    except Exception as e:
        print(f"Error cargando FAISS: {e}")
        return None

    # 3. Configurar el modelo Gemini vía LangChain (v1beta para evitar el 404)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.0,       # 0.0 = sin creatividad, solo hechos
        max_output_tokens=400, # Suficiente para 5 puntos claros
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
        

    def chain(question: str, history_from_app: list = None):
        if not can_call_model():
            return "Por favor, espera un momento..."

        def normalize(q):
            return q.lower().strip()

        if normalize(question) in cache:
            return cache[normalize(question)]
        
        fast = fast_response(question)
        if fast:
            return fast

        try:
            fecha_actual = datetime.datetime.now().strftime("%B %Y")
            periodo_actual = get_current_period()
        # A. Recuperación de documentos
            docs = retriever.invoke(question)
            
            context_docs = docs

            def classify_question(question):
                q = question.lower()

                if "beca" in q:
                    return "becas"
                elif "servicio" in q:
                    return "servicio_social"
                elif "estancia" in q:
                    return "estancia_profesional"

                return "general"

            # B. Filtrado por Categoría (usando el nombre del archivo en metadata)
            category_filter =  classify_question(question)
            q_lower = question.lower()
            if "beca" in q_lower: category_filter = "becas"
            elif "servicio" in q_lower: category_filter = "servicio_social"
            elif "estancia" in q_lower: category_filter = "estancia_profesional"

            if category_filter:
                filtered = [d for d in docs if category_filter in d.metadata.get('source', '').lower()]
                if filtered: 
                    context_docs = filtered

            context_text = "\n---\n".join(d.page_content for d in context_docs[:3])

            # C. Historial
            history_text = ""
            if history_from_app:
                history_text = "\n".join([f"Usuario: {h['user']}\nBot: {h['bot']}" for h in history_from_app[-3:]])

            # E. Prompt Maestro
            # QUITA el prompt anterior completo y PON este:
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
# E. Ejecución
            response = llm.invoke(prompt)
            answer = fix_incomplete_answer(response.content)

            cache[normalize(question)] = answer
            return answer

        except Exception as e:
            # ESTO ES LO QUE VERÁS EN LA TERMINAL
            print(f"Error interno en la cadena: {str(e)}")
            return f"Lo siento, hubo un error técnico al procesar la pregunta."
    return chain