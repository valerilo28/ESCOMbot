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
# main.py está en app/, su parent es la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent        # → app/
STATIC_PATH = BASE_DIR / "static"                 # → app/static/
PDF_DIR = BASE_DIR / "data" / "pdfs"              # → app/data/pdfs/
PDF_DIR.mkdir(parents=True, exist_ok=True)

# --- SUPABASE (opcional) ---
try:
    from app.storage.supabase_client import supabase
except ImportError:
    supabase = None

# --- VARIABLE GLOBAL ---
chain = None
startup_failed = False  # True si load_chain() retornó None

# --- MODELOS ---
class ChatRequest(BaseModel):
    question: str
    history: List[dict] = []

# --- STARTUP EN BACKGROUND ---
async def _load_chain_background():
    global chain
    try:
        print("[STARTUP] Iniciando carga de chain...")
        import importlib
        mod = importlib.import_module("app.rag.chain")
        load_chain = mod.load_chain

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, load_chain)
        if result is None:
            print("[STARTUP] ❌ load_chain() retornó None.")
        else:
            chain = result
            print("[STARTUP] ✅ Chain lista.")
    except Exception:
        import traceback
        print(f"[STARTUP ERROR]\n{traceback.format_exc()}")

# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Usar ensure_future en lugar de create_task para mayor compatibilidad
    asyncio.ensure_future(_load_chain_background())
    yield

# --- APP ---
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
    return {
        "status": "ok",
        "chain_loaded": chain is not None,
        "groq_key": bool(os.getenv("GROQ_API_KEY")),
        "hf_token": bool(os.getenv("HF_TOKEN"))
    }

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

        # 1. Guardar en disco primero
        file_path = PDF_DIR / filename
        with open(file_path, "wb") as f:
            f.write(content)

        # 2. Subir a Supabase
        if supabase:
            try:
                supabase.storage.from_("pdfs").upload(filename, content)
            except Exception as e_supa:
                print(f"[SUPABASE] Error subiendo: {e_supa}")

        # 3. Reconstruir vectorstore con el nuevo PDF ya en disco
        loop = asyncio.get_event_loop()
        from app.rag.vectorstore import build_vectorstore
        await loop.run_in_executor(None, build_vectorstore)

        # 4. Recargar chain
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
        waited = 0
        while chain is None and waited < 120:
            await asyncio.sleep(3)
            waited += 3
            print(f"[CHAT] Esperando chain... {waited}s")

        if chain is None:
            return {"answer": "El servidor tardó demasiado en iniciar. Intenta de nuevo.", "status": "error"}

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: chain(request.question, request.history)
        )
        return {"answer": response, "status": "ok"}

    except Exception as e:
        import traceback
        print(f"[CHAT ERROR]\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error procesando la pregunta")

@app.post("/chat/temario")
async def chat_temario(request: ChatRequest):
    """Endpoint exclusivo para consultas de temarios y bibliografía."""
    try:
        waited = 0
        while chain is None and waited < 120:
            await asyncio.sleep(3)
            waited += 3

        if chain is None:
            return {"answer": "El servidor tardó demasiado en iniciar. Intenta de nuevo.", "status": "error"}

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: chain(request.question, request.history, force_category="temario")
        )
        return {"answer": response, "status": "ok"}

    except Exception as e:
        import traceback
        print(f"[CHAT/TEMARIO ERROR]\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error procesando la pregunta")

@app.get("/suggestions")
async def get_suggestions():
    return {
        "suggestions": [
            "¿Cómo solicito una beca?",
            "Requisitos para servicio social",
            "¿Qué bibliografía tiene Cálculo Diferencial?",
            "¿Cómo contacto a Gestión Escolar?"
        ]
    }
