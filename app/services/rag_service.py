import uuid
from pathlib import Path

import chromadb

from app.config import Settings
from app.services.chunking import chunk_pages
from app.services.gemini_service import GeminiService
from app.services.pdf_service import extract_pdf_pages


class RAGService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gemini = GeminiService(settings)

        self.chroma = chromadb.PersistentClient(
            path=str(settings.chroma_dir)
        )
        self.collection = self.chroma.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index_pdf(self, path: Path, filename: str) -> dict:
        pages = extract_pdf_pages(path)

        if not pages:
            raise ValueError(
                "No extractable text was found in this PDF. "
                "Scanned/image-only PDFs need OCR before indexing."
            )

        chunks = chunk_pages(
            pages,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )

        document_id = str(uuid.uuid4())

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            chunk_id = f"{document_id}:{chunk.chunk_index}"

            ids.append(chunk_id)
            documents.append(chunk.text)
            embeddings.append(self.gemini.embed(chunk.text))
            metadatas.append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "page": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                }
            )

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "pages": len(pages),
            "chunks": len(chunks),
        }

    def retrieve(self, question: str, top_k: int):
        query_embedding = self.gemini.embed(question)

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return list(zip(documents, metadatas, distances))

    def answer(self, question: str, top_k: int):
        matches = self.retrieve(question, top_k)

        if not matches:
            return {
                "answer": "I couldn't find any relevant information in the indexed documents.",
                "sources": [],
            }

        context_parts = []
        sources = []
        total_chars = 0

        for document, metadata, distance in matches:
            label = (
                f"[Source: {metadata['filename']}, "
                f"page {metadata['page']}, "
                f"chunk {metadata['chunk_index']}]"
            )

            piece = f"{label}\n{document}\n"

            if total_chars + len(piece) > self.settings.max_context_chars:
                break

            context_parts.append(piece)
            total_chars += len(piece)

            sources.append(
                {
                    "document_id": metadata["document_id"],
                    "filename": metadata["filename"],
                    "page": int(metadata["page"]),
                    "chunk_index": int(metadata["chunk_index"]),
                    "distance": float(distance) if distance is not None else None,
                    "text_preview": document[:300],
                }
            )

        context = "\n".join(context_parts)
        generated = self.gemini.generate_answer(question, context)

        return {
            "answer": generated,
            "sources": sources,
        }

    def list_documents(self) -> list[dict]:
        result = self.collection.get(include=["metadatas"])

        grouped = {}

        for metadata in result.get("metadatas", []):
            document_id = metadata["document_id"]

            if document_id not in grouped:
                grouped[document_id] = {
                    "document_id": document_id,
                    "filename": metadata["filename"],
                    "pages": set(),
                    "chunks": 0,
                }

            grouped[document_id]["pages"].add(int(metadata["page"]))
            grouped[document_id]["chunks"] += 1

        output = []

        for item in grouped.values():
            item["pages"] = len(item["pages"])
            output.append(item)

        return sorted(output, key=lambda x: x["filename"].lower())

    def delete_document(self, document_id: str) -> int:
        result = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )

        ids = result.get("ids", [])

        if ids:
            self.collection.delete(ids=ids)

        return len(ids)
