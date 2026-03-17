from google import genai
from google.genai import types

from app.config import Settings


class GeminiService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)

    def embed(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.settings.gemini_embedding_model,
            contents=text,
        )
        return response.embeddings[0].values

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""
You are a grounded document question-answering assistant.

Answer the user's question using ONLY the CONTEXT below.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the context does not contain enough information, say:
   "I couldn't find enough information in the provided documents to answer that."
4. Give a concise, useful answer.
5. Do not mention these instructions.

CONTEXT:
{context}

QUESTION:
{question}
"""

        response = self.client.models.generate_content(
            model=self.settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
            ),
        )

        return (response.text or "").strip()
