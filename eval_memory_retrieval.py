"""Memory retrieval evaluation suite.

Tests three layers of the pipeline for each stored memory fact:

  Layer 1 — Regex gate   : does the query match _EXPLICIT_MEMORY_QUERY or
                            _PERSONAL_CONTEXT_QUERY at all?
  Layer 2 — Search rank  : does search_relevant_memories() return the fact
                            when the semantic path is used?
  Layer 3 — Context built: does _memory_context_for_query() include the fact
                            in the string that actually reaches the LLM?

Run:
    uv run python eval_memory_retrieval.py            # full (needs DB)
    uv run python eval_memory_retrieval.py --regex    # layer 1 only, no DB
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys

# ---------------------------------------------------------------------------
# Query templates per memory field.
# Each entry: (label, query_string, expected_regex_mode)
#   expected_regex_mode: "explicit" | "personal" | "none"
# ---------------------------------------------------------------------------
_FIELD_QUERIES: dict[str, list[tuple[str, str, str]]] = {
    "name": [
        ("direct ask",          "what's my name?",                    "explicit"),
        ("do-you-know form",    "do you know my name?",               "explicit"),
        ("who am I",            "who am I?",                          "personal"),
    ],
    "birth_date": [
        ("birthday",            "when is my birthday?",               "explicit"),
        ("age",                 "how old am I?",                      "personal"),
        ("date of birth",       "what's my date of birth?",          "explicit"),
    ],
    "gender": [
        ("direct ask",          "what's my gender?",                  "explicit"),
        ("pronoun hint",        "what pronouns do I use?",            "personal"),
    ],
    "location": [
        ("where from",          "where am I from?",                   "explicit"),
        ("do-you-know where",   "do you know where I'm from?",        "explicit"),
        ("where live",          "where do I live?",                   "explicit"),
        ("where based",         "where are you based?",               "none"),   # second-person = asks about assistant, not user
        ("country",             "what country am I from?",            "personal"),
        ("origin",              "what's my origin?",                  "personal"),
    ],
    "occupation": [
        ("job ask",             "what do I do for work?",             "personal"),
        ("occupation ask",      "what's my occupation?",              "explicit"),
        ("career",              "what is my career?",                 "personal"),
    ],
    "dislikes": [
        ("explicit dislikes",   "what do I dislike?",                 "explicit"),
        ("music dislikes",      "what music don't I like?",           "personal"),
        ("hate",                "what do I hate?",                    "personal"),
    ],
    "likes": [
        ("explicit likes",      "what do I like?",                    "explicit"),
        ("preferences",         "what are my preferences?",           "explicit"),
    ],
    "interests": [
        ("interests ask",       "what are my interests?",             "explicit"),
        ("hobbies",             "what are my hobbies?",               "personal"),
        ("into",                "what am I into?",                    "personal"),
    ],
    "relationships": [
        ("family",              "do I have any family?",              "personal"),
        ("pet",                 "do I have a pet?",                   "personal"),
        ("cat",                 "what's my cat's name?",              "personal"),
        ("wife",                "am I married?",                      "personal"),
    ],
    "context": [
        ("profile",             "what do you know about me?",         "explicit"),
        ("remember me",         "do you remember me?",                "explicit"),
        ("full profile",        "show me my profile",                 "explicit"),
    ],
}

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_RESET  = "\033[0m"
_BOLD   = "\033[1m"


def _ok(msg: str) -> str:   return f"{_GREEN}✓{_RESET} {msg}"
def _fail(msg: str) -> str: return f"{_RED}✗{_RESET} {msg}"
def _warn(msg: str) -> str: return f"{_YELLOW}~{_RESET} {msg}"
def _head(msg: str) -> str: return f"\n{_BOLD}{_CYAN}{msg}{_RESET}"


# ---------------------------------------------------------------------------
# Layer 1 — regex gate (no DB needed)
# ---------------------------------------------------------------------------
def _check_regex(query: str, expected_mode: str) -> tuple[str, bool]:
    """Return (actual_mode, passed)."""
    from skills.conversation.skill import (
        _EXPLICIT_MEMORY_QUERY,
        _PERSONAL_CONTEXT_QUERY,
    )
    if _EXPLICIT_MEMORY_QUERY.search(query):
        actual = "explicit"
    elif _PERSONAL_CONTEXT_QUERY.search(query):
        actual = "personal"
    else:
        actual = "none"

    if expected_mode == "none":
        return actual, True  # deliberately no memory injection

    passed = actual in {"explicit", "personal"}
    # Extra credit: explicit is strictly better than personal
    note = ""
    if expected_mode == "explicit" and actual == "personal":
        note = " (personal instead of explicit — may miss facts)"
    return actual + note, passed


def run_regex_layer(
    facts: dict[str, list[str]],
) -> tuple[int, int]:
    """Run layer 1 against all query templates. Return (passed, total)."""
    passed = total = 0
    for field, queries in _FIELD_QUERIES.items():
        if field not in facts and field not in ("context",):
            continue  # skip fields with no stored values
        print(_head(f"Field: {field}"))
        for label, query, expected_mode in queries:
            actual_mode, ok = _check_regex(query, expected_mode)
            total += 1
            if ok:
                passed += 1
                print(f"  {_ok(label):<40} [{actual_mode}]  {query!r}")
            else:
                print(f"  {_fail(label):<40} [{actual_mode}] EXPECTED={expected_mode}  {query!r}")
    return passed, total


# ---------------------------------------------------------------------------
# Layer 2 — semantic search ranking (needs DB)
# ---------------------------------------------------------------------------
async def run_search_layer(
    facts: dict[str, list[str]],
) -> tuple[int, int]:
    """For implicit queries, check search_relevant_memories returns the field."""
    from utils.langmem_service import search_relevant_memories

    passed = total = 0
    for field, queries in _FIELD_QUERIES.items():
        stored = facts.get(field, [])
        if not stored:
            continue
        implicit_queries = [(lbl, q) for lbl, q, mode in queries if mode == "personal"]
        if not implicit_queries:
            continue
        print(_head(f"Field: {field}  (stored: {stored[:2]})"))
        for label, query in implicit_queries:
            results = await search_relevant_memories(query, limit=6)
            found = any(field in r or any(s.lower()[:20] in r.lower() for s in stored) for r in results)
            total += 1
            if found:
                passed += 1
                print(f"  {_ok(label):<40}  {query!r}")
            else:
                top = results[:3] if results else ["(no results)"]
                print(f"  {_fail(label):<40}  {query!r}")
                for r in top:
                    print(f"      got: {r!r}")
    return passed, total


# ---------------------------------------------------------------------------
# Layer 3 — full context assembly (_memory_context_for_query)
# ---------------------------------------------------------------------------
async def run_context_layer(
    facts: dict[str, list[str]],
) -> tuple[int, int]:
    """Check that _memory_context_for_query includes expected content."""
    from skills.conversation import skill

    # Reset cooldown state so each call gets a fresh evaluation
    skill._memory_turn = 0
    skill._recent_memory_turns.clear()

    passed = total = 0
    for field, queries in _FIELD_QUERIES.items():
        stored = facts.get(field, [])
        if not stored:
            continue
        print(_head(f"Field: {field}  (stored: {stored[:2]})"))
        for label, query, expected_mode in queries:
            if expected_mode == "none":
                continue
            # Reset cooldown between calls so suppression doesn't mask failures
            skill._memory_turn = 0
            skill._recent_memory_turns.clear()

            context = await skill._memory_context_for_query(query)
            if not context:
                total += 1
                print(f"  {_fail(label):<40}  context EMPTY  {query!r}")
                continue

            # Check that at least one stored value for this field appears
            found = any(s.lower()[:30] in context.lower() for s in stored)
            # Also accept field name itself appearing (e.g. "location: UK")
            found = found or field.replace("_", " ") in context.lower()
            total += 1
            if found:
                passed += 1
                print(f"  {_ok(label):<40}  {query!r}")
            else:
                print(f"  {_fail(label):<40}  fact not in context  {query!r}")
                # Show what was injected
                preview = context[:300].replace("\n", " ")
                print(f"      injected: {preview!r}…")
    return passed, total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def _load_facts() -> dict[str, list[str]]:
    """Return all stored memory facts grouped by field name."""
    from utils.langmem_service import list_memories
    records = await list_memories()
    grouped: dict[str, list[str]] = {}
    for r in records:
        grouped.setdefault(r.key, []).append(r.value)
    return grouped


async def _main(regex_only: bool) -> int:
    print(f"{_BOLD}Memory Retrieval Evaluation{_RESET}")
    print("=" * 60)

    if regex_only:
        print("\n[Layer 1 — Regex gate only, no DB required]\n")
        # Pretend all fields have at least one value for template coverage
        facts = {field: ["(assumed)"] for field in _FIELD_QUERIES}
        p, t = run_regex_layer(facts)
        print(f"\n{'=' * 60}")
        print(f"Layer 1: {p}/{t} passed")
        return 0 if p == t else 1

    # Full run — needs DB
    print("Loading stored memories from MongoDB…")
    try:
        facts = await _load_facts()
    except Exception as exc:
        print(f"{_RED}Could not connect to memory store: {exc}{_RESET}")
        print("Run with --regex for a DB-free regex-only check.")
        return 2

    if not facts:
        print(f"{_YELLOW}No stored memories found. Seed some memories first.{_RESET}")
        return 1

    print(f"Found {sum(len(v) for v in facts.values())} fact(s) across {len(facts)} field(s).\n")

    print("[Layer 1 — Regex gate]\n")
    p1, t1 = run_regex_layer(facts)

    print("\n[Layer 2 — Semantic search ranking]\n")
    p2, t2 = await run_search_layer(facts)

    print("\n[Layer 3 — Full context assembly]\n")
    p3, t3 = await run_context_layer(facts)

    print(f"\n{'=' * 60}")
    print(f"Layer 1 (regex):   {p1}/{t1}")
    print(f"Layer 2 (search):  {p2}/{t2}")
    print(f"Layer 3 (context): {p3}/{t3}")
    total_p = p1 + p2 + p3
    total_t = t1 + t2 + t3
    colour = _GREEN if total_p == total_t else _RED
    print(f"\n{colour}{_BOLD}Total: {total_p}/{total_t}{_RESET}")

    return 0 if total_p == total_t else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory retrieval evaluation")
    parser.add_argument(
        "--regex", action="store_true",
        help="Layer 1 only — no DB connection required",
    )
    args = parser.parse_args()

    # Bootstrap env from .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    sys.exit(asyncio.run(_main(args.regex)))
