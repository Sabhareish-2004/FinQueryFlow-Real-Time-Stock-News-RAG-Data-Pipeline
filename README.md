"""
FinQueryFlow - Data Ingestion Pipeline
Fetches real-time stock news from NewsAPI, Polygon, Reddit
Pushes raw articles to Kafka for downstream processing
"""

import asyncio
import json
import httpx
from datetime import datetime, timedelta
from kafka import KafkaProducer
from typing import List, Dict, Any
from models.schemas import RawArticle
import os

KAFKA_BROKER   = os.getenv("KAFKA_BROKER",   "localhost:9092")
NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY",    "YOUR_KEY")
POLYGON_KEY    = os.getenv("POLYGON_KEY",    "YOUR_KEY")
REDDIT_CLIENT  = os.getenv("REDDIT_CLIENT",  "YOUR_CLIENT")
REDDIT_SECRET  = os.getenv("REDDIT_SECRET",  "YOUR_SECRET")

WATCHLIST = ["AAPL", "TSLA", "GOOGL", "MSFT", "NVDA", "AMZN", "META", "AMD"]


def get_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


async def fetch_newsapi(ticker: str, client: httpx.AsyncClient) -> List[Dict]:
    """Fetch news articles from NewsAPI for a given ticker."""
    since = (datetime.utcnow() - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%S")
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={ticker}&from={since}&sortBy=publishedAt"
        f"&language=en&apiKey={NEWSAPI_KEY}"
    )
    resp = await client.get(url, timeout=10)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])
    return [
        {
            "source":     "newsapi",
            "ticker":     ticker,
            "title":      a["title"],
            "description":a.get("description",""),
            "url":        a["url"],
            "published_at": a["publishedAt"],
            "content":    a.get("content",""),
        }
        for a in articles if a.get("title")
    ]


async def fetch_polygon(ticker: str, client: httpx.AsyncClient) -> List[Dict]:
    """Fetch news from Polygon.io for a given ticker."""
    url = (
        f"https://api.polygon.io/v2/reference/news"
        f"?ticker={ticker}&limit=20&order=desc&apiKey={POLYGON_KEY}"
    )
    resp = await client.get(url, timeout=10)
    if resp.status_code != 200:
        return []
    results = resp.json().get("results", [])
    return [
        {
            "source":     "polygon",
            "ticker":     ticker,
            "title":      r["title"],
            "description": r.get("description",""),
            "url":        r["article_url"],
            "published_at": r["published_utc"],
            "content":    r.get("description",""),
        }
        for r in results
    ]


async def fetch_reddit_sentiment(ticker: str, client: httpx.AsyncClient) -> List[Dict]:
    """Fetch top Reddit posts mentioning a ticker from r/wallstreetbets."""
    headers = {"User-Agent": "FinQueryFlow/1.0"}
    url = f"https://www.reddit.com/r/wallstreetbets/search.json?q={ticker}&sort=new&limit=10"
    resp = await client.get(url, headers=headers, timeout=10)
    if resp.status_code != 200:
        return []
    posts = resp.json().get("data", {}).get("children", [])
    return [
        {
            "source":      "reddit",
            "ticker":      ticker,
            "title":       p["data"]["title"],
            "description": p["data"].get("selftext","")[:500],
            "url":         f"https://reddit.com{p['data']['permalink']}",
            "published_at":datetime.utcfromtimestamp(p["data"]["created_utc"]).isoformat(),
            "content":     p["data"].get("selftext",""),
            "upvotes":     p["data"].get("ups",0),
        }
        for p in posts
    ]


async def ingest_all(tickers: List[str] = WATCHLIST):
    """Main ingestion loop: fetch all sources, push to Kafka."""
    producer = get_producer()
    async with httpx.AsyncClient() as client:
        tasks = []
        for ticker in tickers:
            tasks.append(fetch_newsapi(ticker, client))
            tasks.append(fetch_polygon(ticker, client))
            tasks.append(fetch_reddit_sentiment(ticker, client))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        total = 0
        for batch in results:
            if isinstance(batch, Exception):
                print(f"[Ingest] Error: {batch}")
                continue
            for article in batch:
                producer.send("raw-news", value=article)
                total += 1
        producer.flush()
        print(f"[Ingest] Pushed {total} articles to Kafka at {datetime.utcnow().isoformat()}")


if __name__ == "__main__":
    asyncio.run(ingest_all())
