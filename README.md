# Agentic AI RAG Chatbot

A Python-based Retrieval-Augmented Generation (RAG) chatbot built for the AI Engineering Intern technical assignment.

The chatbot answers questions strictly from the **Agentic AI eBook** provided as the knowledge base. It uses **LangGraph** for the RAG workflow, **Pinecone** for vector search, **Sentence Transformers** for text embeddings, **Gemini** for answer generation, and **FastAPI** for the API.

## Features

- PDF ingestion and text extraction
- Page-aware text chunking
- Sentence Transformer embeddings
- Pinecone vector database
- LangGraph RAG pipeline
- Gemini LLM for grounded answer generation
- Retrieval relevance threshold for out-of-scope questions
- Retrieved context chunks included in the API response
- Retrieval similarity score included in the API response
- FastAPI REST API
- Swagger/OpenAPI documentation
- Simple Python implementation

## Architecture

```text
                  Agentic AI PDF
                       |
                       v
                PDF Text Extraction
                       |
                       v
                 Text Chunking
                       |
                       v
            Sentence Transformer
                 Embeddings
                       |
                       v
              Pinecone Vector DB
                       |
                       |
User Question ---------+
       |
       v
Question Embedding
       |
       v
Pinecone Similarity Search
       |
       v
 Top Relevant Chunks
       |
       v
 Relevance Threshold
       |
       +---- Low Score ----> "Information not found"
       |
       v
    LangGraph
       |
       v
   Gemini LLM
       |
       v
 Grounded Answer
       |
       v
    FastAPI
       |
       v
Answer + Context + Score
```

## Tech Stack

- **Python 3.10+**
- **LangGraph** — RAG workflow orchestration
- **Pinecone** — vector database and similarity search
- **Sentence Transformers** — text embeddings
- **Gemini** — LLM for answer generation
- **FastAPI** — REST API
- **PyPDF** — PDF text extraction
- **LangChain** — LLM and text-processing integration

## Project Structure

```text
agentic-ai-rag/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── graph.py
│   ├── main.py
│   ├── retriever.py
│   └── schemas.py
│
├── ingestion/
│   ├── __init__.py
│   └── ingest.py
│
├── data/
│   └── .gitkeep
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Knowledge Base

The project uses the Agentic AI eBook provided in the assignment:

**Agentic AI — An Executive's Guide to In-depth Understanding of Agentic AI**

Source:

https://konverge.ai/pdf/Ebook-Agentic-AI.pdf

The PDF is intentionally not committed to the repository. It should be downloaded locally into the `data/` directory before running ingestion.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd agentic-ai-rag
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
GEMINI_API_KEY=your_gemini_api_key
```

API keys must never be committed to GitHub.

### 5. Add the PDF

Download the Agentic AI eBook from the assignment source and save it as:

```text
data/Ebook-Agentic-AI.pdf
```

### 6. Run ingestion

From the project root:

```bash
python -m ingestion.ingest
```

The ingestion process:

1. Extracts text from the PDF.
2. Preserves the source page number.
3. Splits text into overlapping chunks.
4. Generates embeddings using Sentence Transformers.
5. Creates the Pinecone index if it does not already exist.
6. Uploads embeddings and metadata to Pinecone.

The current dataset contains approximately **59 pages and 137 text chunks** after ingestion.

## Run the API

Start FastAPI with:

```bash
uvicorn app.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## API

### `GET /`

Health check endpoint.

Example response:

```json
{
  "message": "Agentic AI RAG Chatbot API is running"
}
```

### `POST /chat`

Ask a question about the Agentic AI eBook.

Request:

```json
{
  "question": "What is Agentic AI?"
}
```

Example response:

```json
{
  "answer": "Based on the provided context, Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives.",
  "confidence": 0.8169,
  "retrieved_context": [
    {
      "page": 18,
      "score": 0.7558,
      "text": "Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives..."
    }
  ]
}
```

## Response Fields

### `answer`

The final answer generated from the retrieved PDF context.

### `confidence`

The highest similarity score among the retrieved Pinecone chunks.

> This is a **retrieval similarity score**, not a calibrated probability or statistical confidence value.

### `retrieved_context`

The chunks retrieved from the PDF. Each item contains:

- `page` — source PDF page
- `score` — Pinecone similarity score
- `text` — retrieved chunk

## Grounding and Out-of-Scope Questions

The chatbot is designed to avoid answering questions using information outside the provided PDF.

The retrieval stage first checks the highest similarity score against a minimum relevance threshold.

If the retrieved score is below the threshold, the system returns:

```text
I couldn't find this information in the provided PDF.
```

For relevant questions, the retrieved chunks are passed to Gemini with an explicit instruction to answer only from the supplied context and not use outside knowledge.

This creates the following flow:

```text
Question
   |
   v
Embedding
   |
   v
Pinecone Retrieval
   |
   v
Relevance Check
   |
   +---- Not Relevant ----> Fallback Response
   |
   v
Retrieved Context
   |
   v
Gemini
   |
   v
Grounded Answer
```

## Sample Queries

### 1. What is Agentic AI?

Tests the basic definition of Agentic AI.

### 2. How does Agentic AI differ from other AI?

Tests the discussion about how Agentic AI stands apart from other AI approaches.

### 3. What can Agentic AI do?

Tests retrieval of Agentic AI capabilities.

### 4. What value does Agentic AI bring to businesses?

Tests business value and organizational applications discussed in the eBook.

### 5. How are businesses using Agentic AI?

Tests retrieval of real-world business use cases.

### 6. What is the capital of France?

This is intentionally outside the knowledge base.

Expected response:

```text
I couldn't find this information in the provided PDF.
```

This demonstrates the grounding behavior.

## LangGraph Workflow

The LangGraph implementation contains two main nodes:

```text
START
  |
  v
Retrieve
  |
  v
Generate
  |
  v
END
```

### Retrieve Node

The user's question is converted into an embedding using the Sentence Transformer model.

Pinecone performs cosine-similarity search and returns the top relevant chunks together with their metadata and scores.

### Generate Node

The retrieved chunks are assembled into a context and passed to Gemini.

The prompt instructs the model to:

- use only the supplied context
- avoid outside knowledge
- avoid making up information
- return the fallback response when the information is not available

## Embedding Model

The project uses:

```text
all-MiniLM-L6-v2
```

The embedding dimension is:

```text
384
```

The embeddings are stored in Pinecone using cosine similarity.

## Chunking

The PDF is split using `RecursiveCharacterTextSplitter`.

Current configuration:

```text
chunk_size = 800
chunk_overlap = 150
```

Page numbers are stored as metadata so that retrieved information can be traced back to the source PDF.

## Security

- API keys are stored in `.env`.
- `.env` is excluded through `.gitignore`.
- The local PDF is excluded from Git tracking.
- No API credentials are stored in source code.

## Running the Complete Pipeline

### First-time setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`, download the PDF, then:

```bash
python -m ingestion.ingest
```

Start the API:

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

Use `POST /chat` to query the knowledge base.

## Example Questions

```text
What is Agentic AI?

How does Agentic AI differ from other AI?

What can Agentic AI do?

What value does Agentic AI bring to businesses?

How are businesses using Agentic AI?

What is the capital of France?
```

## Assignment Deliverable

This repository implements the requested:

- PDF ingestion
- Chunking
- Text embeddings
- Pinecone vector storage
- LangGraph RAG pipeline
- LLM-based grounded generation
- FastAPI chat API
- Retrieved context output
- Retrieval score output
- Sample queries
- Architecture and setup documentation

Built as a hands-on Python implementation for the **AI Engineering Intern** technical assignment.
