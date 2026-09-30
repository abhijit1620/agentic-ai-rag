from fastapi import FastAPI, HTTPException

from app.graph import rag_graph
from app.schemas import QueryRequest, QueryResponse


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description="RAG chatbot based on the Agentic AI ebook",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Agentic AI RAG Chatbot API is running"
    }


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        result = rag_graph.invoke({
            "question": question,
            "contexts": [],
            "scores": [],
            "answer": ""
        })

        retrieved_context = []

        for context, score in zip(
            result.get("contexts", []),
            result.get("scores", [])
        ):
            retrieved_context.append({
                "page": context["page"],
                "score": round(float(score), 4),
                "text": context["text"]
            })

        scores = result.get("scores", [])

        confidence = (
            round(float(max(scores)), 4)
            if scores
            else 0.0
        )

        return {
            "answer": result["answer"],
            "confidence": confidence,
            "retrieved_context": retrieved_context
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {str(e)}"
        )
