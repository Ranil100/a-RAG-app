from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str
    gemini_model: str = "gemini-2.5-flash"
    gemini_embedding_model: str = "gemini-embedding-001"

    chroma_path: str = "./storage/chroma"
    collection_name: str = "rag_documents"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    max_context_chars: int = 12000

    @property
    def chroma_dir(self) -> Path:
        path = Path(self.chroma_path)
        if not path.is_absolute():
            path = BASE_DIR / path
        return path

    @property
    def uploads_dir(self) -> Path:
        return BASE_DIR / "data" / "uploads"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    return settings
