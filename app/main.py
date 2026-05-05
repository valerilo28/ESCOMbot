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

# --- DIRECTORIOS ---
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_PATH = BASE_DIR / "app" / "static"
PDF_DIR = BASE_DIR / "data" / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# --- SUPABASE (opcional) ---
try:
    from app.storage.supabase_client import supabase
except ImportError:
    supabase = None

# --- VARIABLE GLOBAL ---
chain = None

# --- MODELOS ---
class ChatRequest(BaseModel):
    question: str
    history: List[dict] = []

# --- STARTUP EN BACKGROUND ---
async def _load_chain_background():
    global chain
    loop = asyncio.get_event_loop()
    try:
        print("[STARTUP] Paso 1: Descargando PDFs...")
        from app.storage.download_pdfs import download_pdfs
        await loop.run_in_executor(None, download_pdfs)

        print("[STARTUP] Paso 2: Construyendo FAISS...")
        from app.rag.vectorstore import build_vectorstore
        await loop.run_in_executor(None, build_vectorstore)

        print("[STARTUP] Paso 3: Cargando chain...")
        from app.rag.chain import load_chain
        chain = await loop.run_in_executor(None, load_chain)
        print("[STARTUP] ✅ Chain lista.")

    except Exception as e:
        import traceback
        print(f"[STARTUP ERROR]\n{traceback.format_exc()}")

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_load_chain_background())
    yield

# --- APP (debe crearse ANTES de cualquier @app.ruta) ---
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

# --- RUTAS ---

@app.get("/health")
async def health():
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
        filename = f"{category}_{year}-{semester}_{file.filename.replace(' ', '_')}"

        file_path = PDF_DIR / filename
        with open(file_path, "wb") as f:
            f.write(content)

        if supabase:
            try:
                supabase.storage.from_("pdfs").upload(filename, content)
            except Exception as e_supa:
                print(f"[SUPABASE] Error subiendo: {e_supa}")

        loop = asyncio.get_event_loop()
        from app.rag.vectorstore import build_vectorstore
        await loop.run_in_executor(None, build_vectorstore)

        from app.rag.chain import load_chain
        chain = load_chain()

        return {"message": f"Archivo {filename} cargado con éxito."}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if not chain:
            return {"answer": "El sistema se está iniciando, espera un momento... ⏳"}

        response = chain(request.question, request.history)
        return {"answer": response, "status": "ok"}

    except Exception as e:
        import traceback
        print(f"[CHAT ERROR]\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error procesando la pregunta")

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