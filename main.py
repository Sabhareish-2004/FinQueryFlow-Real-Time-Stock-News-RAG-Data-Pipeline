"""
FinQueryFlow - Sentiment Analysis Route
Uses VADER for quick scoring + FinBERT for finance-specific sentiment
Aggregates per-ticker sentiment from the last N hours
"""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional, List
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline as hf_pipeline
import re

router = APIRouter()
PG_DSN = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/finqueryflow")

vader = SentimentIntensityAnalyzer()

# FinBERT — load lazily to avoid cold-start weight
_finbert = None
def get_finbert():
    global _finbert
    if _finbert is None:
        _finbert = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
    return _finbert


class SentimentScore(BaseModel):
    ticker: str
    vader_compound: float
    finbert_label: str
    finbert_score: float
    article_count: int
    bullish_pct: float
    bearish_pct: float
    neutral_pct: float
    sample_headlines: List[str]


def score_vader(text: str) -> float:
    return vader.polarity_scores(text)["compound"]


def score_finbert(texts: List[str]) -> dict:
    fb = get_finbert()
    results = fb(texts[:8])  # limit batch to 8 for speed
    counts = {"positive":0, "negative":0, "neutral":0}
    total_score = 0.0
    for r in results:
        label = r["label"].lower()
        counts[label] = counts.get(label, 0) + 1
        score = r["score"] if label == "positive" else -r["score"] if label == "negative" else 0
        total_score += score
    n = len(results)
    dominant = max(counts, key=counts.get)
    return {
        "label": dominant,
        "score": round(total_score / n, 4) if n else 0,
        "bullish_pct": round(counts["positive"] / n * 100, 1) if n else 0,
        "bearish_pct": round(counts["negative"] / n * 100, 1) if n else 0,
        "neutral_pct": round(counts["neutral"]  / n * 100, 1) if n else 0,
    }


@router.get("/ticker/{ticker}", response_model=SentimentScore)
async def ticker_sentiment(
    ticker: str,
    hours: int = Query(24, ge=1, le=168),
):
    """
    Aggregate sentiment for a ticker over the last N hours.
    Combines VADER compound score with FinBERT label.
    """
    conn = psycopg2.connect(PG_DSN)
    with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT title FROM articles
            WHERE ticker = %s
              AND published_at > NOW() - INTERVAL '%s hours'
            ORDER BY published_at DESC
            LIMIT 50
        """, (ticker.upper(), hours))
        rows = cur.fetchall()
    conn.close()

    if not rows:
        return SentimentScore(
            ticker=ticker.upper(), vader_compound=0.0, finbert_label="neutral",
            finbert_score=0.0, article_count=0, bullish_pct=0, bearish_pct=0,
            neutral_pct=100, sample_headlines=[],
        )

    headlines = [r["title"] for r in rows]
    vader_scores = [score_vader(h) for h in headlines]
    avg_vader = round(sum(vader_scores) / len(vader_scores), 4)
    fb = score_finbert(headlines)

    return SentimentScore(
        ticker           =ticker.upper(),
        vader_compound   =avg_vader,
        finbert_label    =fb["label"],
        finbert_score    =fb["score"],
        article_count    =len(headlines),
        bullish_pct      =fb["bullish_pct"],
        bearish_pct      =fb["bearish_pct"],
        neutral_pct      =fb["neutral_pct"],
        sample_headlines =headlines[:5],
    )


@router.get("/watchlist")
async def watchlist_sentiment(
    tickers: str = Query("AAPL,TSLA,GOOGL,MSFT,NVDA"),
    hours: int = Query(24),
):
    """Batch sentiment for a comma-separated list of tickers."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    from fastapi.concurrency import run_in_threadpool
    results = []
    for ticker in ticker_list[:10]:  # max 10 for speed
        result = await ticker_sentiment(ticker, hours=hours)
        results.append(result)
    return {"results": results, "hours": hours}
