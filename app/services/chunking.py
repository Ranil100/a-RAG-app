from dataclasses import dataclass


@dataclass
class TextChunk:
    text: str
    page_number: int
    chunk_index: int


def chunk_pages(
    pages,
    chunk_size: int,
    overlap: int,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    chunks: list[TextChunk] = []
    global_index = 0

    for page in pages:
        text = " ".join(page.text.split())

        if not text:
            continue

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()

            if piece:
                chunks.append(
                    TextChunk(
                        text=piece,
                        page_number=page.page_number,
                        chunk_index=global_index,
                    )
                )
                global_index += 1

            if end >= len(text):
                break

            start = end - overlap

    return chunks
