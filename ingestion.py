"""
FinQueryFlow - News API Route
Serves paginated news articles from PostgreSQL with filtering by ticker/source/date
"""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime, timedelta

router = APIRouter()
PG_DSN = os.getenv("DATABASE_URL", "postgresql://finuser:finpass@localhost:5432/finqueryflow")


class Article(BaseModel):
    id:           str
    ticker:       str
    source:       str
    title:        str
    url:          Optional[str]
    published_at: Optional[str]
    ingested_at:  Optional[str]


class NewsPage(BaseModel):
    articles: List[Article]
    total:    int
    page:     int
    pages:    int


@router.get("/feed", response_model=NewsPage)
async def get_news_feed(
    ticker:  Optional[str] = Query(None, description="Filter by ticker symbol e.g. NVDA"),
    source:  Optional[str] = Query(None, description="Filter by source: newsapi | polygon | reddit"),
    hours:   int           = Query(24, ge=1, le=168, description="Lookback window in hours"),
    page:    int           = Query(1, ge=1),
    limit:   int           = Query(20, ge=1, le=100),
):
    """
    Paginated news feed from PostgreSQL.
    Supports filtering by ticker, source, and lookback window.
    """
    offset = (page - 1) * limit
    since  = datetime.utcnow() - timedelta(hours=hours)
    conditions = ["published_at > %s"]
    params: list = [since]

    if ticker:
        conditions.append("ticker = %s")
        params.append(ticker.upper())
    if source:
        conditions.append("source = %s")
        params.append(source.lower())

    where = " AND ".join(conditions)

    try:
        conn = psycopg2.connect(PG_DSN)
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT COUNT(*) as cnt FROM articles WHERE {where}", params)
            total = cur.fetchone()["cnt"]

            cur.execute(
                f"""
                SELECT id, ticker, source, title, url,
                       published_at::text, ingested_at::text
                FROM articles WHERE {where}
                ORDER BY published_at DESC
                LIMIT %s OFFSET %s
                """,
                params + [limit, offset],
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

    import math
    return NewsPage(
        articles=[Article(**r) for r in rows],
        total=total,
        page=page,
        pages=math.ceil(total / limit),
    )


@router.get("/latest/{ticker}")
async def latest_news(ticker: str, limit: int = Query(10, ge=1, le=50)):
    """Get the N most recent articles for a ticker."""
    try:
        conn = psycopg2.connect(PG_DSN)
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, ticker, source, title, url, published_at::text
                FROM articles WHERE ticker = %s
                ORDER BY published_at DESC LIMIT %s
                """,
                (ticker.upper(), limit),
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"ticker": ticker.upper(), "articles": rows, "count": len(rows)}


@router.get("/stats")
async def news_stats():
    """Summary statistics: article counts by ticker and source for last 24h."""
    try:
        conn = psycopg2.connect(PG_DSN)
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT ticker, source, COUNT(*) as count
                FROM articles
                WHERE published_at > NOW() - INTERVAL '24 hours'
                GROUP BY ticker, source
                ORDER BY count DESC
            """)
            rows = cur.fetchall()

            cur.execute("SELECT COUNT(*) as total FROM articles")
            total = cur.fetchone()["total"]
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"breakdown": rows, "total_articles_all_time": total}
