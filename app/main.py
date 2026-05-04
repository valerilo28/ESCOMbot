import os
import asyncio
from pathlib import Path
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Intentamos importar la lógica del chatbot
try:
    from app.rag.chain import load_chain
except ModuleNotFoundError:
    from rag.chain import load_chain

# Intentamos importar Supabase
try:
    from app.storage.supabase_client import supabase
except ImportError:
    supabase = None # Por si aún no configuras el cliente

# 1. INICIALIZAR LA APP (Primero que nada)
app = FastAPI(title="Chatbot ESCOM - Gemini")

# 2. CONFIGURAR RUTAS Y DIRECTORIOS
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_PATH = BASE_DIR / "app" /"static"
PDF_DIR = BASE_DIR / "app" / "data" / "pdfs"

# Crear carpetas si no existen
PDF_DIR.mkdir(parents=True, exist_ok=True)

# 3. MONTAR ARCHIVOS ESTÁTICOS (Después de crear 'app')
if STATIC_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

# Variables globales
chain = None

def get_chain():
    global chain
    if chain is None:
        print("[LOG] Cargando chain por demanda...")
        chain = load_chain()
    return chain

# Modelos de datos
class ChatRequest(BaseModel):
    question: str
    history: List[dict] = [] # Añadido para soportar historial desde la app

# --- EVENTOS ---
@app.on_event("startup")
async def startup_event():
    global chain
    # Cargamos el modelo en segundo plano para no bloquear el inicio del servidor
    loop = asyncio.get_event_loop()
    chain = await loop.run_in_executor(None, load_chain)
    print("Chatbot listo para recibir preguntas.")

# --- RUTAS DE ADMINISTRACIÓN ---

@app.get("/upload", response_class=HTMLResponse)
async def get_upload_page():
    html_file = STATIC_PATH / "index.html"
    if not html_file.exists():
        return f"<h1>Error: No se encontró index.html en {STATIC_PATH}</h1>"
    return html_file.read_text(encoding="utf-8")

@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    category: str = Form(...),
    year: str = Form(...),
    semester: str = Form(...)
):
    try:
        content = await file.read()
        
        # Estructura de nombre: categoria_año-semestre_nombreoriginal.pdf
        # Ejemplo: becas_2026-2_convocatoria.pdf
        filename = f"{category}_{year}-{semester}_{file.filename.replace(' ', '_')}"
        
        # 1. Guardar localmente (Necesario para que FAISS lo procese luego)
        file_path = PDF_DIR / filename
        with open(file_path, "wb") as f:
            f.write(content)
        
        # 2. Subir a Supabase (Opcional, según tu flujo)
        if supabase:
            try:
                supabase.storage.from_("pdfs").upload(filename, content)
            except Exception as e_supa:
                print(f"Error subiendo a Supabase: {e_supa}")

        return {"message": f"Archivo {filename} cargado y clasificado con éxito."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- RUTAS DE LA APP MÓVIL ---

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        current_chain = get_chain()
        if not current_chain:
            return {"answer": "El sistema se está iniciando, por favor espera un momento..."}

        # Pasamos la pregunta y el historial (si tu chain lo soporta)
        response = current_chain(request.question, request.history)
        return {
            "answer": response,
            "status": "ok"
        }

    except Exception as e:
        print(f"Error en chat: {e}")
        return {"error": "Hubo un error al procesar tu pregunta."}

@app.get("/suggestions")
async def get_suggestions():
    return {
        "suggestions": [
            "¿Cómo solicito una beca?",
            "Requisitos para servicio social",
            "¿Cuándo son los exámenes ETS?",
            "¿Cómo contacto a Gestión Escolar?"
        ]
    }

# 4. CONFIGURAR CORS (Al final para asegurar que todas las rutas lo tengan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)