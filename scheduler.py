"""
FinQueryFlow - RAG API Route
Exposes /api/rag/ask endpoint with optional streaming
"""

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from rag.engine import ask, retrieve, format_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
import json

router = APIRouter()

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "YOUR_KEY")


class AskRequest(BaseModel):
    question: str
    ticker: Optional[str] = None
    stream: bool = False


class AskResponse(BaseModel):
    answer: str
    sources: list
    cached: bool


@router.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """
    RAG Q&A endpoint.
    - Retrieves top-k news chunks from ChromaDB
    - Generates a grounded answer using GPT-4o-mini
    - Returns answer + source citations
    """
    result = await ask(req.question, ticker=req.ticker, stream=req.stream)
    return result


@router.get("/ask/stream")
async def ask_stream(
    question: str = Query(...),
    ticker: Optional[str] = Query(None),
):
    """
    Streaming RAG endpoint using Server-Sent Events.
    Frontend subscribes to this for real-time token output.
    """
    from rag.engine import retrieve, format_context, prompt, llm
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnablePassthrough
    from datetime import datetime

    chunks = retrieve(question, ticker_filter=ticker)
    context = format_context(chunks)

    async def event_stream():
        chain = (
            {"context": lambda _: context, "question": RunnablePassthrough(), "today": lambda _: datetime.utcnow().strftime("%Y-%m-%d")}
            | prompt
            | llm
            | StrOutputParser()
        )
        async for token in chain.astream(question):
            yield f"data: {json.dumps({'token': token})}\n\n"

        sources = [
            {"ticker":c["ticker"],"source":c["source"],"url":c["url"],"score":c["score"]}
            for c in chunks
        ]
        yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/sources")
async def get_sources(
    question: str = Query(...),
    ticker: Optional[str] = Query(None),
    top_k: int = Query(5),
):
    """Preview which chunks would be retrieved for a given question (debug tool)."""
    chunks = retrieve(question, ticker_filter=ticker, top_k=top_k)
    return {"chunks": chunks, "count": len(chunks)}
