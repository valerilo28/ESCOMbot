import os
import asyncio
from pathlib import Path
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from app.rag.vectorstore import build_vectorstore

# Intentamos importar la lógica del chatbot
try:
    from app.rag.chain import load_chain
except ModuleNotFoundError:
    from rag.chain import load_chain

# Intentamos importar Supabase
try:
    from app.storage.supabase_client import supabase
except ImportError:
    supabase = None

# --- DIRECTORIOS ---
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_PATH = BASE_DIR / "app" / "static"
PDF_DIR = BASE_DIR / "app" / "data" / "pdfs"

PDF_DIR.mkdir(parents=True, exist_ok=True)

# --- VARIABLE GLOBAL ---
chain = None

def get_chain():
    global chain
    if chain is None:
        print("[LOG] Cargando chain por demanda...")
        chain = load_chain()
    return chain

# --- MODELOS ---
class ChatRequest(BaseModel):
    question: str
    history: List[dict] = []

# --- LIFESPAN (carga en background para no bloquear el puerto) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain
    # Cargamos en background para que Render detecte el puerto de inmediato
    asyncio.create_task(_load_chain_background())
    yield

async def _load_chain_background():
    global chain
    loop = asyncio.get_event_loop()
    print("[STARTUP] Cargando chain en background...")
    chain = await loop.run_in_executor(None, load_chain)
    print("[STARTUP] Chain listo.")

# --- APP (una sola instancia, con lifespan y CORS desde el inicio) ---
app = FastAPI(title="Chatbot ESCOM", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

# --- RUTAS DE ADMINISTRACIÓN ---

@app.get("/health")
async def health():
    """Endpoint para que la app móvil detecte si el servidor está activo."""
    return {"status": "ok", "chain_loaded": chain is not None}

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
    global chain
    try:
        content = await file.read()

        # Estructura: categoria_año-semestre_nombreoriginal.pdf
        filename = f"{category}_{year}-{semester}_{file.filename.replace(' ', '_')}"

        # 1. Guardar localmente PRIMERO
        file_path = PDF_DIR / filename
        with open(file_path, "wb") as f:
            f.write(content)

        # 2. Subir a Supabase
        if supabase:
            try:
                supabase.storage.from_("pdfs").upload(filename, content)
            except Exception as e_supa:
                print(f"Error subiendo a Supabase: {e_supa}")

        # 3. Reconstruir vectorstore con el nuevo PDF ya en disco
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, build_vectorstore)

        # 4. Recargar el chain con el índice actualizado
        chain = load_chain()

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
