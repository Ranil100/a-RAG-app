from pathlib import Path

from app.config import get_settings
from app.services.rag_service import RAGService


settings = get_settings()
rag = RAGService(settings)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def upload_pdf():
    print_header("PDF DOCUMENT INGESTION")

    file_input = input("Enter PDF path: ").strip().strip('"')

    if not file_input:
        print("❌ No file path entered.")
        return

    pdf_path = Path(file_input)

    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return

    if pdf_path.suffix.lower() != ".pdf":
        print("❌ Please provide a PDF file.")
        return

    print("\n📄 Reading PDF...")
    print("✂️ Splitting text into chunks...")
    print("🧠 Creating embeddings with Gemini...")
    print("🗄️ Storing vectors in ChromaDB...")

    try:
        result = rag.index_pdf(pdf_path, pdf_path.name)

        print("\n✅ Document indexed successfully!")
        print(f"   Document : {result['filename']}")
        print(f"   Pages    : {result['pages']}")
        print(f"   Chunks   : {result['chunks']}")
        print(f"   ID       : {result['document_id']}")

    except Exception as exc:
        print(f"\n❌ Indexing failed: {exc}")


def ask_question():
    print_header("RAG QUESTION ANSWERING")

    question = input("❓ Your question: ").strip()

    if not question:
        print("❌ Please enter a question.")
        return

    print("\n🔎 Searching and generating answer...")

    try:
        top_k = settings.top_k

        print("🧠 Creating question embedding...")
        matches = rag.retrieve(question, top_k)

        if not matches:
            print("\n❌ No relevant chunks found.")
            return

        print(f"🔍 Found {len(matches)} relevant chunks:\n")

        for index, (_, metadata, distance) in enumerate(matches, start=1):

            # ChromaDB cosine distance:
            # lower distance = more similar
            similarity = max(
                0.0,
                min(1.0, 1.0 - float(distance))
            )

            print(
                f"   Chunk {index}: "
                f"Score {similarity:.3f} "
                f"(~Page {metadata['page']})"
            )

            print(
                f"            {metadata['filename']}"
            )

        print("\n📚 Building context from retrieved chunks...")

        context_parts = []
        total_chars = 0
        sources = []

        for document, metadata, distance in matches:

            label = (
                f"[Source: {metadata['filename']}, "
                f"page {metadata['page']}, "
                f"chunk {metadata['chunk_index']}]"
            )

            piece = f"{label}\n{document}\n"

            if (
                total_chars + len(piece)
                > settings.max_context_chars
            ):
                break

            context_parts.append(piece)
            total_chars += len(piece)

            sources.append(
                {
                    "filename": metadata["filename"],
                    "page": metadata["page"],
                    "chunk_index": metadata["chunk_index"],
                }
            )

        context = "\n".join(context_parts)

        print("🤖 Sending retrieved context to Gemini...")

        answer = rag.gemini.generate_answer(
            question,
            context
        )

        print_header("ANSWER")

        print(answer)

        print("\n📌 Sources:")

        for source in sources:
            print(
                f"   - {source['filename']} "
                f"(Page {source['page']}, "
                f"Chunk {source['chunk_index']})"
            )

    except Exception as exc:
        print(f"\n❌ RAG query failed: {exc}")


def list_documents():
    print_header("INDEXED DOCUMENTS")

    try:
        documents = rag.list_documents()

        if not documents:
            print("No documents have been indexed yet.")
            return

        for index, document in enumerate(
            documents,
            start=1
        ):
            print(f"\n{index}. {document['filename']}")
            print(f"   Pages : {document['pages']}")
            print(f"   Chunks: {document['chunks']}")
            print(f"   ID    : {document['document_id']}")

    except Exception as exc:
        print(
            f"❌ Could not list documents: {exc}"
        )


def delete_document():
    print_header("DELETE DOCUMENT")

    documents = rag.list_documents()

    if not documents:
        print("No documents available.")
        return

    for index, document in enumerate(
        documents,
        start=1
    ):
        print(
            f"{index}. {document['filename']}"
        )

    choice = input(
        "\nEnter document number: "
    ).strip()

    try:
        index = int(choice) - 1
        document = documents[index]

    except (ValueError, IndexError):
        print("❌ Invalid selection.")
        return

    deleted = rag.delete_document(
        document["document_id"]
    )

    print(
        f"\n✅ Deleted {deleted} chunks belonging to "
        f"{document['filename']}."
    )


def main():

    print_header(
        "INTELLIGENT DOCUMENT Q&A — RAG"
    )

    print(
        "Terminal mode enabled."
    )

    print(
        "You can test the complete RAG pipeline "
        "from this terminal."
    )

    while True:

        print("\n")
        print("1. Upload PDF")
        print("2. Ask Question")
        print("3. List Documents")
        print("4. Delete Document")
        print("5. Exit")

        choice = input(
            "\nChoose an option: "
        ).strip()

        if choice == "1":
            upload_pdf()

        elif choice == "2":
            ask_question()

        elif choice == "3":
            list_documents()

        elif choice == "4":
            delete_document()

        elif choice == "5":
            print(
                "\n👋 Exiting RAG application."
            )
            break

        else:
            print(
                "❌ Invalid option. Choose 1-5."
            )


if __name__ == "__main__":
    main()