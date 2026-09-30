from pathlib import Path
import time

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
)


PDF_PATH = Path("data/Ebook-Agentic-AI.pdf")


def load_pdf():
    reader = PdfReader(PDF_PATH)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def create_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "page": page["page"]
            })

    return chunks


def create_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = pc.list_indexes().names()

    if PINECONE_INDEX_NAME not in existing_indexes:
        print("Creating Pinecone index...")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            print("Waiting for Pinecone index...")
            time.sleep(2)

    return pc.Index(PINECONE_INDEX_NAME)


def upload_chunks(chunks, index):
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    vectors = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding.tolist(),
            "metadata": {
                "text": chunk["text"],
                "page": chunk["page"],
                "source": "Ebook-Agentic-AI.pdf"
            }
        })

    print("Uploading vectors to Pinecone...")

    index.upsert(
        vectors=vectors
    )

    print(f"Uploaded {len(vectors)} vectors.")


if __name__ == "__main__":
    pages = load_pdf()
    chunks = create_chunks(pages)

    print(f"Pages extracted: {len(pages)}")
    print(f"Created chunks: {len(chunks)}")

    index = create_pinecone_index()

    upload_chunks(chunks, index)

    print("Ingestion completed successfully.")