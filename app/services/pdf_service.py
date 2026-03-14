from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class PageText:
    page_number: int
    text: str


def extract_pdf_pages(path: Path) -> list[PageText]:
    pages: list[PageText] = []

    with fitz.open(path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append(PageText(page_number=index + 1, text=text))

    return pages
