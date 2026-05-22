import os
import asyncio
from pathlib import Path
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager

BASE_DIR = Path(__file__).resolve().parent
STATIC_PATH = BASE_DIR / "static"
PDF_DIR = BASE_DIR / "data" / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)

try:
    from app.storage.supabase_client import supabase
except ImportError:
    supabase = None

chain = None
chain_loading = False   # flag para saber si está en proceso
chain_error = None      # guardar error de startup si ocurre

class ChatRequest(BaseModel):
    question: str
    history: List[dict] = []

class LoginRequest(BaseModel):
    username: str
    password: str

async def _load_chain_background():
    global chain, chain_loading, chain_error
    chain_loading = True
    chain_error = None
    try:
        print("[STARTUP] Cargando chain...")
        import importlib
        mod = importlib.import_module("app.rag.chain")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, mod.load_chain)
        if result is None:
            chain_error = "load_chain() retornó None"
            print(f"[STARTUP] ❌ {chain_error}")
        else:
            chain = result
            print("[STARTUP] ✅ Chain lista.")
    except Exception:
        import traceback
        chain_error = traceback.format_exc()
        print(f"[STARTUP ERROR]\n{chain_error}")
    finally:
        chain_loading = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.ensure_future(_load_chain_background())
    yield

app = FastAPI(title="Chatbot ESCOM", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

if STATIC_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chain_loaded": chain is not None,
        "chain_loading": chain_loading,
        "chain_error": chain_error,
        "groq_key": bool(os.getenv("GROQ_API_KEY")),
    }

@app.post("/login")
async def login(request: LoginRequest):
    admin_users = os.getenv("ADMIN_USERS", "admin@escom.ipn.mx,2024630001").split(",")
    admin_password = os.getenv("ADMIN_PASSWORD", "escom2026")
    username = request.username.strip().lower()
    if username not in [u.strip().lower() for u in admin_users] or request.password != admin_password:
        raise HTTPException(status_code=401, detail="Credenciales incorrectas.")
    return {"status": "ok", "message": "Acceso concedido."}

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
    semester: str = Form(...),
    password: str = Form(...)
):
    if password != os.getenv("ADMIN_PASSWORD", "escom2026"):
        raise HTTPException(status_code=401, detail="Contraseña incorrecta.")
    try:
        content = await file.read()
        filename = f"{category}_{year}-{semester}_{file.filename.replace(' ', '_')}"
        file_path = PDF_DIR / filename
        with open(file_path, "wb") as f:
            f.write(content)
        try:
            from app.storage.supabase_client import get_supabase
            get_supabase().storage.from_("pdfs").upload(filename, content)
        except Exception as e:
            print(f"[SUPABASE] ⚠️ {e}")
        return {"message": f"✅ {filename} guardado. Reconstruye el índice con rebuild_index.py y haz push."}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Esperar máx 60s (no 120s) con mensajes más claros
        waited = 0
        max_wait = 60
        while chain is None and waited < max_wait:
            if not chain_loading and chain_error:
                # El startup falló — no tiene caso seguir esperando
                return JSONResponse(
                    status_code=503,
                    content={"answer": "El servidor tuvo un error al iniciar. Contacta al administrador.", "status": "error"}
                )
            await asyncio.sleep(2)
            waited += 2
            if waited % 10 == 0:
                print(f"[CHAT] Esperando chain... {waited}s/{max_wait}s")

        if chain is None:
            return JSONResponse(
                status_code=503,
                content={
                    "answer": "El servidor está iniciando, por favor recarga la página en unos segundos. 🔄",
                    "status": "loading"
                }
            )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: chain(request.question, request.history)
        )
        return {"answer": response, "status": "ok"}

    except Exception:
        import traceback
        print(f"[CHAT ERROR]\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error procesando la pregunta")

@app.post("/chat/temario")
async def chat_temario(request: ChatRequest):
    try:
        waited = 0
        while chain is None and waited < 60:
            await asyncio.sleep(2)
            waited += 2
        if chain is None:
            return {"answer": "El servidor está iniciando. Recarga en unos segundos. 🔄", "status": "loading"}
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: chain(request.question, request.history, force_category="temario")
        )
        return {"answer": response, "status": "ok"}
    except Exception:
        import traceback
        print(f"[CHAT/TEMARIO ERROR]\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error procesando la pregunta")

@app.get("/status")
async def status():
    """Endpoint liviano para que el frontend sepa si la chain ya está lista."""
    return {
        "ready": chain is not None,
        "loading": chain_loading,
        "error": chain_error is not None
    }

@app.get("/suggestions")
async def get_suggestions():
    return {
        "suggestions": [
            "¿Cómo solicito una beca?",
            "Requisitos para servicio social",
            "¿Qué bibliografía tiene Cálculo Diferencial?",
            "¿Qué es estancia profesional?"
        ]
    }