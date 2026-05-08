import difflib
import json
import os
import re
import time
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
            data = ticker.history(period=period)
            return data.to_string()
        else:
            data = ticker.fast_info['lastPrice']
            return str(data)
    except Exception as error:
        return error


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
