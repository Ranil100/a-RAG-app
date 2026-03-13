from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_index: int
    distance: float | None = None
    text_preview: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    message: str


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
