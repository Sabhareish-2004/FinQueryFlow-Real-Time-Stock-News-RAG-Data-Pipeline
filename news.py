"""
FinQueryFlow - RAG Engine
Semantic retrieval from ChromaDB + Reranking + LLM answer generation via LangChain
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import chromadb
from chromadb.config import Settings
import redis
import json
import hashlib
import os
from typing import Optional
from datetime import datetime

OPENAI_KEY  = os.getenv("OPENAI_API_KEY", "YOUR_KEY")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8001"))
REDIS_URL   = os.getenv("REDIS_URL", "redis://localhost:6379")

CACHE_TTL_SECONDS = 300  # 5 min cache for repeated queries

# ----- Clients -----

def get_collection():
    client = chromadb.HttpClient(
        host=CHROMA_HOST, port=CHROMA_PORT,
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection("stock_news")


embedder   = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_KEY)
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_KEY,
    temperature=0.1,
    streaming=True,
)

# ----- Prompt -----

SYSTEM_PROMPT = """You are FinQueryFlow, an expert financial analyst assistant.
Answer questions grounded strictly on the provided news context.
Be concise, cite sources when relevant, and flag uncertainty clearly.
Do NOT fabricate data. If context is insufficient, say so honestly.

Context documents:
{context}

Today's date: {today}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


# ----- Retrieval -----

def retrieve(
    question: str,
    ticker_filter: Optional[str] = None,
    top_k: int = 8,
) -> list[dict]:
    """Embed query and retrieve top-k chunks from ChromaDB with optional ticker filter."""
    collection = get_collection()
    query_vec = embedder.embed_query(question)

    where = {"ticker": ticker_filter} if ticker_filter else None
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        chunks.append({
            "text":         doc,
            "ticker":       meta.get("ticker",""),
            "source":       meta.get("source",""),
            "url":          meta.get("url",""),
            "published_at": meta.get("published_at",""),
            "score":        round(1 - dist, 4),   # cosine similarity
        })

    # Simple rerank: prefer recent high-score chunks
    chunks.sort(key=lambda c: (c["score"], c["published_at"]), reverse=True)
    return chunks[:5]


def format_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        lines.append(
            f"[{i}] ({c['ticker']} | {c['source']} | {c['published_at'][:10]})\n{c['text']}"
        )
    return "\n\n".join(lines)


# ----- Cache -----

def cache_key(question: str, ticker: Optional[str]) -> str:
    raw = f"{question}|{ticker or 'all'}"
    return "rag:" + hashlib.md5(raw.encode()).hexdigest()


# ----- Main RAG Function -----

async def ask(
    question: str,
    ticker: Optional[str] = None,
    stream: bool = False,
) -> dict:
    """
    Full RAG pipeline:
      1. Check Redis cache
      2. Retrieve relevant chunks from ChromaDB
      3. Generate grounded answer with GPT-4o-mini
      4. Cache and return result with sources
    """
    key = cache_key(question, ticker)
    cached = redis_client.get(key)
    if cached:
        result = json.loads(cached)
        result["cached"] = True
        return result

    chunks = retrieve(question, ticker_filter=ticker)
    if not chunks:
        return {
            "answer":  "No relevant news found in the knowledge base for your query.",
            "sources": [],
            "cached":  False,
        }

    context = format_context(chunks)
    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough(), "today": lambda _: datetime.utcnow().strftime("%Y-%m-%d")}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = await chain.ainvoke(question)

    result = {
        "answer":  answer,
        "sources": [{"ticker":c["ticker"],"source":c["source"],"url":c["url"],"score":c["score"],"date":c["published_at"][:10]} for c in chunks],
        "cached":  False,
    }
    redis_client.setex(key, CACHE_TTL_SECONDS, json.dumps(result))
    return result
