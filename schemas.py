"""
FinQueryFlow - Stocks API Route
Fetches OHLCV price history from Yahoo Finance via yfinance
"""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

router = APIRouter()


class OHLCV(BaseModel):
    date:   str
    open:   float
    high:   float
    low:    float
    close:  float
    volume: int


class StockInfo(BaseModel):
    ticker:        str
    name:          str
    sector:        Optional[str]
    market_cap:    Optional[float]
    pe_ratio:      Optional[float]
    week_52_high:  Optional[float]
    week_52_low:   Optional[float]
    current_price: Optional[float]
    change_pct:    Optional[float]


@router.get("/price/{ticker}", response_model=List[OHLCV])
async def get_price_history(
    ticker: str,
    period: str = Query("1mo", enum=["1d","5d","1mo","3mo","6mo","1y","2y","5y"]),
    interval: str = Query("1d", enum=["1m","5m","15m","1h","1d","1wk","1mo"]),
):
    """
    Return OHLCV price history for a ticker.
    Uses yfinance as the data source.
    """
    try:
        df = yf.download(ticker.upper(), period=period, interval=interval, progress=False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Yahoo Finance error: {e}")

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {ticker}")

    df = df.reset_index()
    records = []
    for _, row in df.iterrows():
        records.append(OHLCV(
            date   =str(row.get("Date", row.get("Datetime",""))),
            open   =round(float(row["Open"]),   2),
            high   =round(float(row["High"]),   2),
            low    =round(float(row["Low"]),    2),
            close  =round(float(row["Close"]),  2),
            volume =int(row["Volume"]),
        ))
    return records


@router.get("/info/{ticker}", response_model=StockInfo)
async def get_stock_info(ticker: str):
    """Return key fundamental data for a ticker."""
    try:
        t = yf.Ticker(ticker.upper())
        info = t.info
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    hist = t.history(period="2d")
    change_pct = None
    if len(hist) >= 2:
        prev_close = float(hist["Close"].iloc[-2])
        last_close = float(hist["Close"].iloc[-1])
        change_pct = round((last_close - prev_close) / prev_close * 100, 2)

    return StockInfo(
        ticker        =ticker.upper(),
        name          =info.get("longName", ticker),
        sector        =info.get("sector"),
        market_cap    =info.get("marketCap"),
        pe_ratio      =info.get("trailingPE"),
        week_52_high  =info.get("fiftyTwoWeekHigh"),
        week_52_low   =info.get("fiftyTwoWeekLow"),
        current_price =info.get("currentPrice") or info.get("regularMarketPrice"),
        change_pct    =change_pct,
    )


@router.get("/watchlist")
async def watchlist_snapshot(
    tickers: str = Query("AAPL,TSLA,GOOGL,MSFT,NVDA,AMZN,META,AMD"),
):
    """Return a compact snapshot of price and change for a list of tickers."""
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    results = []
    for ticker in ticker_list[:15]:
        try:
            info = await get_stock_info(ticker)
            results.append(info)
        except Exception:
            pass
    return {"results": results}
