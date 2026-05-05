import os
import asyncio   
import traceback
from pathlib import Path
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager

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
PDF_DIR = BASE_DIR / "data" / "pdfs"

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

# main.py — reemplaza _load_chain_background completo
async def _load_chain_background():
    global chain
    loop = asyncio.get_event_loop()
    try:
        print("[STARTUP] Paso 1: Descargando PDFs desde Supabase...")
        from app.storage.download_pdfs import download_pdfs
        await loop.run_in_executor(None, download_pdfs)

        print("[STARTUP] Paso 2: Construyendo índice FAISS...")
        from app.rag.vectorstore import build_vectorstore
        await loop.run_in_executor(None, build_vectorstore)

        print("[STARTUP] Paso 3: Cargando chain...")
        chain = await loop.run_in_executor(None, load_chain)
        print("[STARTUP] Todo listo.")
    except Exception as e:
        import traceback
        print(f"[STARTUP ERROR]\n{traceback.format_exc()}")
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
