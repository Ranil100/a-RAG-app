# Intelligent Document Q&A API — RAG Backend

A backend-only Retrieval-Augmented Generation (RAG) mini project built with FastAPI, ChromaDB, PyMuPDF, and Google's Gemini API.

## Features

- Upload PDF documents
- Extract text from PDFs
- Split text into overlapping chunks
- Generate Gemini embeddings
- Store embeddings in ChromaDB
- Semantic similarity retrieval
- Generate grounded answers with Gemini
- Return source document/page/chunk metadata
- List and delete indexed documents
- Swagger/OpenAPI documentation
- Configurable top-k retrieval and chunking

## Architecture

PDF -> Text Extraction -> Chunking -> Gemini Embeddings -> ChromaDB
                                                        |
Question -> Gemini Embedding -> Similarity Search -> Context -> Gemini -> Answer

## Requirements

- Python 3.10+
- A Google Gemini API key

## Setup

### 1. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env.example` to `.env` and add your API key:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
CHROMA_PATH=./storage/chroma
COLLECTION_NAME=rag_documents
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
TOP_K=5
MAX_CONTEXT_CHARS=12000
```

### 4. Start the API

```bash
uvicorn app.main:app --reload
```

Open:

- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc
- Health: http://127.0.0.1:8000/health

## API Endpoints

### GET /health

Checks whether the backend is running.

### POST /documents/upload

Upload a PDF using multipart form-data.

### GET /documents

Lists indexed documents.

### DELETE /documents/{document_id}

Deletes all chunks belonging to a document.

### POST /ask

Example request:

```json
{
  "question": "What is the main purpose of this document?",
  "top_k": 5
}
```

The response includes the generated answer and retrieved sources.

## Example curl

Upload:

```bash
curl -X POST "http://127.0.0.1:8000/documents/upload" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@sample.pdf"
```

Ask:

```bash
curl -X POST "http://127.0.0.1:8000/ask" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"Summarize the document\",\"top_k\":5}"
```

## RAG behavior

The answer-generation prompt explicitly instructs Gemini to use only the retrieved context. If the answer is not supported by the indexed documents, the API returns a grounded "I don't know" style response rather than asking the model to invent information.

## Project structure

```text
rag-backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── documents.py
│   │   └── chat.py
│   └── services/
│       ├── __init__.py
│       ├── pdf_service.py
│       ├── chunking.py
│       ├── gemini_service.py
│       └── rag_service.py
├── data/
│   └── uploads/
├── storage/
│   └── chroma/
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Notes

- The project stores ChromaDB data locally in `storage/chroma`.
- Uploaded PDFs are stored in `data/uploads`.
- Do not commit `.env`, API keys, PDFs, or the local vector database to Git.
- For production, add authentication, file-size limits, rate limiting, background ingestion, persistent cloud storage, and an external vector database.

## Portfolio description

**Intelligent Document Q&A API using RAG** — Built a backend RAG system with FastAPI that extracts PDF content, chunks documents, generates semantic embeddings, stores vectors in ChromaDB, retrieves relevant context, and generates grounded answers using Gemini with source/page metadata.
