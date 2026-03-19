from pathlib import Path
import shutil
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import get_settings
from app.schemas import DocumentInfo, UploadResponse
from app.services.rag_service import RAGService


router = APIRouter(prefix="/documents", tags=["Documents"])

settings = get_settings()
rag = RAGService(settings)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    safe_name = Path(file.filename).name
    temp_path = settings.uploads_dir / f"{uuid.uuid4()}_{safe_name}"

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = rag.index_pdf(temp_path, safe_name)

        return UploadResponse(
            **result,
            message="Document indexed successfully.",
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document indexing failed: {exc}",
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@router.get("", response_model=list[DocumentInfo])
def list_documents():
    return rag.list_documents()


@router.delete("/{document_id}")
def delete_document(document_id: str):
    deleted = rag.delete_document(document_id)

    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found.")

    return {
        "document_id": document_id,
        "deleted_chunks": deleted,
        "message": "Document deleted successfully.",
    }
