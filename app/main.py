from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.rag.chain import load_chain
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Chatbot ESCOM - Gemini")

chain = load_chain()

class ChatRequest(BaseModel):
    question: str

class SuggestionResponse(BaseModel):
    suggestions: List[str]

@app.post("/chat")
async def chat(request: ChatRequest):
    response = chain(request.question)
    return {"answer": response}

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