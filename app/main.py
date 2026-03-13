from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router


app = FastAPI(
    title="Intelligent Document Q&A API",
    description=(
        "Backend-only Retrieval-Augmented Generation API using "
        "Gemini embeddings, ChromaDB, and Gemini generation."
    ),
    version="1.0.0",
)

app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "service": "rag-backend",
    }


@app.get("/", tags=["System"])
def root():
    return {
        "message": "Intelligent Document Q&A API is running.",
        "docs": "/docs",
    }
