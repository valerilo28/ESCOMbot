import os
import time
import datetime
from pathlib import Path
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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


# --- CARGA DE LA CADENA ---

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
    llm = ChatOllama(
    #model="llama3",
    model="mistral",
    temperature=0.1,
    num_predict=300
    )
    
    cache = {}

    

    # --- FUNCIÓN INTERNA DE PROCESAMIENTO ---
    
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

            context_text = "\n---\n".join(d.page_content for d in context_docs[:2])

            # C. Historial
            history_text = ""
            if history_from_app:
                history_text = "\n".join([f"Usuario: {h['user']}\nBot: {h['bot']}" for h in history_from_app[-3:]])

            # E. Prompt Maestro
            prompt = f"""Eres el Asistente Oficial de la ESCOM IPN.

            Hoy es {fecha_actual} y el ciclo escolar es {periodo_actual}.

        INSTRUCCIONES DE RESPUESTA:
        1. **FILTRO DE VIGENCIA**: Si la información proviene de un documento con un periodo distinto a {periodo_actual} (ej. 2025-1, 2025-2), debes iniciar tu respuesta con: "Nota: Esta información corresponde al periodo [Periodo del archivo] y es solo para referencia."
        2. **FIDELIDAD AL CONTEXTO**: Responde UNICAMENTE con la información proporcionada en el contexto. Si no está ahí, di: "No cuento con esa información específica, te sugiero preguntar en Gestión Escolar."
        3. **FORMATO**: Responde en un máximo de 5 puntos claros. Usa **negritas** para resaltar requisitos o fechas y viñetas para organizar.
        4. **ESTRUCTURA**: Asegúrate de cerrar todas las ideas. No dejes listas ni frases incompletas.
        5. **TONO**: Sé amable, profesional y directo. No menciones "según el PDF" o "en el texto proporcionado".

    CONTEXTO DE LOS DOCUMENTOS:
    {context_text}

    HISTORIAL DE CONVERSACIÓN:
    {history_text}

    PREGUNTA DEL ALUMNO:
    {question}
    """
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