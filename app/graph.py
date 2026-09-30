from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

from app.retriever import search_documents


MIN_RETRIEVAL_SCORE = 0.40


class RAGState(TypedDict):
    question: str
    contexts: list
    scores: list
    answer: str


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


def retrieve(state: RAGState):
    results = search_documents(
        state["question"],
        top_k=4
    )

    contexts = []
    scores = []

    for match in results["matches"]:
        contexts.append({
            "text": match["metadata"]["text"],
            "page": match["metadata"]["page"]
        })

        scores.append(float(match["score"]))

    return {
        "contexts": contexts,
        "scores": scores
    }


def generate_answer(state: RAGState):
    scores = state["scores"]

    if not scores or max(scores) < MIN_RETRIEVAL_SCORE:
        return {
            "answer": "I couldn't find this information in the provided PDF."
        }

    context_text = ""

    for i, context in enumerate(state["contexts"], start=1):
        context_text += (
            f"\n--- Context {i} (Page {context['page']}) ---\n"
            f"{context['text']}\n"
        )

    prompt = f"""
You are a document-based AI assistant.

Answer the user's question ONLY using the information provided
in the context below.

If the answer cannot be found in the context, say:
"I couldn't find this information in the provided PDF."

Do not use outside knowledge.
Do not make up information.

Context:
{context_text}

User question:
{state["question"]}
"""

    response = llm.invoke(prompt)

    answer = response.content

    if isinstance(answer, list):
        answer = "".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
        )

    return {
        "answer": answer
    }


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate_answer)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


rag_graph = build_graph()
