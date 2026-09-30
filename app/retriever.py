from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
)


model = SentenceTransformer(EMBEDDING_MODEL)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def search_documents(question, top_k=4):
    query_embedding = model.encode(question).tolist()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return results


if __name__ == "__main__":
    question = "What is Agentic AI?"

    results = search_documents(question)

    for i, match in enumerate(results["matches"], start=1):
        print(f"\n--- Result {i} ---")
        print("Score:", match["score"])
        print("Page:", match["metadata"]["page"])
        print("Text:")
        print(match["metadata"]["text"])