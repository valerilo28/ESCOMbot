from fastapi import FastAPI
from fastapi import UploadFile
from app.storage.supabase_client import supabase
from pydantic import BaseModel
from typing import List
from app.rag.chain import load_chain
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI(title="Chatbot ESCOM - Gemini")

#chain = load_chain()
chain= None

class ChatRequest(BaseModel):
    question: str

class SuggestionResponse(BaseModel):
    suggestions: List[str]

@app.on_event("startup")
async def startup_event():
    global chain
    loop = asyncio.get_event_loop()
    chain = await loop.run_in_executor(None, load_chain)

@app.post("/chat")
async def chat(request: ChatRequest):
    if chain is None:
        return {"answer": "El modelo aún se está cargando, intenta de nuevo en unos segundos."}
    response = chain(request.question)
    return {"answer": response}

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile):

    content = await file.read()

    supabase.storage.from_("pdfs").upload(
        file.filename,
        content
    )

    return {"message": "PDF subido correctamente"}

@app.get("/suggestions")
async def get_suggestions():
    return {
        "suggestions": [
            "¿Cómo solicito una beca?",
            "Requisitos para dictamen de electivas",
            "¿Cómo bajarme de una materia?",
            "Liberación de servicio social"
        ]
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permite que cualquier dispositivo se conecte
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)