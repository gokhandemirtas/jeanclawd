import difflib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yfinance as yf
from utils.messages import HumanMessage, SystemMessage
from utils.log import logger
from pydantic import BaseModel

from agents.agent_factory import create_llm_agent
from skills.finance.prompts import finance_comment_prompt, get_finance_intent_prompt
from utils.llm_structured_output import generate_structured_output

_tickers_path = Path(__file__).resolve().parent.parent.parent / "fixtures" / "tickers.json"
with open(_tickers_path) as f:
    _tickers_data = json.load(f)

_valid_tickers = {entry["ticker"].upper() for entry in _tickers_data}

# Build a lookup map: lowercase name/ticker → entry for fast candidate search
_ticker_index: dict[str, dict] = {}
for _entry in _tickers_data:
    _ticker_index[_entry["name"].lower()] = _entry
    _ticker_index[_entry["ticker"].lower()] = _entry


def _find_candidates(query: str, n: int = 15) -> list[dict]:
    """Return the top N ticker entries most likely to match the query."""
    tokens = set(re.sub(r"[^a-z0-9 ]", "", query.lower()).split())
    tokens -= {"stock", "price", "share", "shares", "the", "of", "how", "is", "doing", "what"}

    scored: dict[str, tuple[int, dict]] = {}
    for entry in _tickers_data:
        key = entry["ticker"].upper()
        name_lower = entry["name"].lower()
        ticker_lower = entry["ticker"].lower()
        score = 0
        for token in tokens:
            if token in name_lower or token in ticker_lower:
                score += 2
            elif any(difflib.get_close_matches(token, name_lower.split(), n=1, cutoff=0.8)):
                score += 1
        if score > 0:
            scored[key] = (score, entry)

    top = sorted(scored.values(), key=lambda x: -x[0])[:n]
    candidates = [entry for _, entry in top]

    # Always include an exact ticker match if present
    for token in tokens:
        t = token.upper()
        if t in _valid_tickers and not any(e["ticker"].upper() == t for e in candidates):
            candidates.append(_ticker_index[token])

    return candidates or _tickers_data[:n]  # fallback to first N if nothing matched


class FinanceIntentResponse(BaseModel):
    ticker: str
    dataType: str
    period: Optional[str] = None


def _get_finance_agent():
    return create_llm_agent(model_name=os.environ['MODEL_GENERALIST'])


def _classify_intent(query: str) -> dict:
    try:
        candidates = _find_candidates(query)
        logger.debug(f"Finance candidates: {[e['ticker'] for e in candidates]}")
        result = generate_structured_output(
            model_name=os.environ["MODEL_GENERALIST"],
            user_prompt=query,
            system_prompt=get_finance_intent_prompt(candidates),
            pydantic_model=FinanceIntentResponse,
        )

        intent_data = result.model_dump()
        ticker = (intent_data.get("ticker") or "").upper()
        if ticker not in _valid_tickers:
            logger.warning(f"Ticker '{ticker}' not found in tickers.json")
            return {"ticker": "UNKNOWN_TICKER", "dataType": "CURRENT", "period": None, "error": "TICKER_NOT_IN_JSON"}

        return intent_data

    except Exception as e:
        logger.error(f"Failed to classify finance intent. Error: {e}")
        return {"ticker": "UNKNOWN_TICKER", "dataType": "CURRENT", "period": None}


def _comment_on_data(query: str, data) -> str:
    messages = [
        SystemMessage(content=finance_comment_prompt),
        HumanMessage(content=f"User Query: {query}\n\nFinancial Data:\n{data}")
    ]

    agent = _get_finance_agent()
    response = agent.invoke({"messages": messages})
    return response["messages"][-1].content.strip()


def _retrieve_stock_data(ticker_symbol: str, is_historical: bool, period: Optional[str] = None) -> str:
    ticker = yf.Ticker(ticker_symbol)
    try:
        if is_historical:
            effective_period = period or "3mo"
            hist = ticker.history(period=effective_period)
            if hist.empty:
                return f"No historical data available for {ticker_symbol}."
            return _format_historical_summary(ticker_symbol, hist, effective_period)
        else:
            return _format_current_snapshot(ticker_symbol, ticker)
    except Exception as error:
        return str(error)


def _format_current_snapshot(ticker_symbol: str, ticker: yf.Ticker) -> str:
    fi = ticker.fast_info
    price = fi.get("lastPrice") or fi.get("last_price")
    prev_close = fi.get("previousClose") or fi.get("previous_close")
    day_high = fi.get("dayHigh") or fi.get("day_high")
    day_low = fi.get("dayLow") or fi.get("day_low")
    week52_high = fi.get("fiftyTwoWeekHigh") or fi.get("fifty_two_week_high")
    week52_low = fi.get("fiftyTwoWeekLow") or fi.get("fifty_two_week_low")
    market_cap = fi.get("marketCap") or fi.get("market_cap")
    currency = fi.get("currency", "USD")

    last_trade_time = fi.get("lastTradeTime") or fi.get("last_trade_time")
    if last_trade_time:
        try:
            if isinstance(last_trade_time, (int, float)):
                last_trade_time = datetime.fromtimestamp(last_trade_time, tz=timezone.utc)
            elif last_trade_time.tzinfo is None:
                last_trade_time = last_trade_time.replace(tzinfo=timezone.utc)
            as_of = last_trade_time.isoformat(timespec="seconds")
        except (AttributeError, TypeError, ValueError, OSError):
            as_of = str(last_trade_time)
    else:
        as_of = None

    lines = [f"Ticker: {ticker_symbol}"]
    if as_of:
        lines.append(f"Market data as of: {as_of}")
    if price:
        lines.append(f"Current price: {price:.2f} {currency}")
    if prev_close and price:
        change = price - prev_close
        pct = (change / prev_close) * 100
        sign = "+" if change >= 0 else ""
        lines.append(f"Day change: {sign}{change:.2f} ({sign}{pct:.2f}%)")
    if day_high and day_low:
        lines.append(f"Day range: {day_low:.2f} – {day_high:.2f}")
    if week52_high and week52_low:
        lines.append(f"52-week range: {week52_low:.2f} – {week52_high:.2f}")
    if market_cap:
        lines.append(f"Market cap: {_fmt_large(market_cap)} {currency}")
    return "\n".join(lines)


def _format_historical_summary(ticker_symbol: str, hist, period: str) -> str:
    closes = hist["Close"]
    highs = hist["High"]
    lows = hist["Low"]
    volumes = hist["Volume"]

    start_price = closes.iloc[0]
    end_price = closes.iloc[-1]
    pct_change = ((end_price - start_price) / start_price) * 100
    period_high = highs.max()
    period_low = lows.min()
    avg_vol = volumes.mean()

    # Recent 2-week trend vs prior 2 weeks
    if len(closes) >= 28:
        recent = closes.iloc[-14:].mean()
        prior = closes.iloc[-28:-14].mean()
        trend_pct = ((recent - prior) / prior) * 100
        trend = f"+{trend_pct:.1f}%" if trend_pct >= 0 else f"{trend_pct:.1f}%"
        trend_line = f"Recent trend (last 2 wks vs prior 2 wks): {trend}"
    else:
        trend_line = None

    sign = "+" if pct_change >= 0 else ""
    lines = [
        f"Ticker: {ticker_symbol} | Period: {period}",
        f"Start price: {start_price:.2f}",
        f"End price:   {end_price:.2f}",
        f"Change:      {sign}{pct_change:.2f}%",
        f"Period high: {period_high:.2f}",
        f"Period low:  {period_low:.2f}",
        f"Avg daily volume: {_fmt_large(avg_vol)}",
    ]
    if trend_line:
        lines.append(trend_line)

    # Monthly breakdown for periods >= 3mo
    if len(closes) >= 60:
        lines.append("\nMonthly breakdown:")
        monthly = hist["Close"].resample("ME").last()
        monthly_open = hist["Close"].resample("ME").first()
        for date, close in monthly.items():
            open_price = monthly_open.get(date, close)
            m_pct = ((close - open_price) / open_price) * 100
            sign_m = "+" if m_pct >= 0 else ""
            lines.append(f"  {date.strftime('%b %Y')}: {close:.2f} ({sign_m}{m_pct:.1f}%)")

    return "\n".join(lines)


def _fmt_large(n: float) -> str:
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    return f"{n:.0f}"


def handle_finance_stocks(query: str, retry_count: int = 0) -> str:
    """Handle stock prices and financial information."""
    MAX_RETRIES = 3

    intent = _classify_intent(query)
    logger.debug(f"Extracted intent: {intent} from {query} (attempt {retry_count + 1}/{MAX_RETRIES + 1})")

    ticker_symbol = intent.get("ticker")
    data_type = intent.get("dataType")
    period = intent.get("period")
    is_historical = data_type == "HISTORICAL" if data_type else False

    if intent.get("error") == "TICKER_NOT_IN_JSON":
        return "I couldn't resolve the ticker symbol for that company. Please update tickers.json with the correct entry."

    should_retry = ticker_symbol == "UNKNOWN_TICKER" or not ticker_symbol or not data_type
    if should_retry:
        if retry_count < MAX_RETRIES:
            logger.warning(f"Intent classification failed — retrying ({retry_count + 1}/{MAX_RETRIES})")
            return handle_finance_stocks(query, retry_count + 1)
        else:
            logger.error(f"Failed to classify intent after {MAX_RETRIES} retries")
            return "I couldn't determine which stock you're asking about. Please try rephrasing your question."

    try:
        stock_data = _retrieve_stock_data(ticker_symbol, is_historical, period)
    except Exception as error:
        logger.error(f"An error occurred: {error}")
        return "I encountered an error while retrieving stock data. Please try again."

    try:
        comment = _comment_on_data(query, stock_data)
        logger.debug(f"Raw comment: {comment}")
        return comment
    except Exception as error:
        logger.error(f"An error occurred: {error}")
        return "I encountered an error while analyzing the stock data. Please try again."
