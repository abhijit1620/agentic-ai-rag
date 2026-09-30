import os

from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

PINECONE_INDEX_NAME = "agentic-ai-rag"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"