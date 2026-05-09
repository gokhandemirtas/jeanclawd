"""Compact a day's conversation exchanges into a plain-language summary."""

import os
from datetime import date

from agents.persona import AGENT_NAME
from db.schemas import Conversations
from utils.log import logger

_COMPACT_PROMPT = f"""You are summarising a day's conversation log between a user and their personal assistant {AGENT_NAME}.
Write a concise, factual summary (3-8 sentences) in third person covering the key topics discussed,
decisions made, and anything the user expressed interest in or asked to follow up on.
Be specific — include names, numbers, and topics mentioned. Do not add commentary or opinions."""

_COMPACT_APPEND_PROMPT = f"""You are updating a running summary of a day's conversation between a user and their personal assistant {AGENT_NAME}.
You have a previous summary and a full conversation log. Produce a single updated summary (3-10 sentences) that incorporates
any new topics, decisions, or details not already covered by the previous summary.
Keep all existing facts. Add new ones. Do not repeat yourself. Write in third person. No commentary or opinions."""


async def compact_conversations(for_date: date) -> str:
    """Summarise exchanges for the given date, save back to the document, and return the summary.

    If a previous summary exists, merges it with the full log into an updated summary.
    Always resets compact_status to 'error' before re-raising so the endpoint is not
    left stuck in the 'pending' state on any failure path.
    """
    doc = await Conversations.find_one(Conversations.date == for_date)
    if doc is None:
        raise ValueError(f"No conversations document found for {for_date}")
    if not doc.exchanges:
        raise ValueError(f"No exchanges to compact for {for_date}")

    try:
        from skills.personalizer.skill import analyze_persona_from_log
        await analyze_persona_from_log(doc.exchanges)

        lines = [f"Conversation log for {for_date}:"]
        for ex in doc.exchanges:
            ts = ex.timestamp.strftime("%H:%M")
            lines.append(f"[{ts}] User: {ex.question}")
            lines.append(f"[{ts}] {AGENT_NAME}: {ex.answer}")
        log_text = "\n".join(lines)

        from utils.messages import HumanMessage, SystemMessage
        from agents.agent_factory import create_llm

        llm = create_llm(
            model_name=os.environ.get("MODEL_GENERALIST"),
            temperature=0.3,
            max_tokens=512,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1,
        )

        if doc.compacted:
            system = _COMPACT_APPEND_PROMPT
            user_content = f"## Previous summary\n{doc.compacted}\n\n## Full conversation log\n{log_text}"
        else:
            system = _COMPACT_PROMPT
            user_content = log_text

        response = await llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user_content),
        ])
        summary = response.content.strip()

        doc.compacted = summary
        doc.compact_status = "complete"
        await doc.save()
        logger.info(f"[compact] Compacted {len(doc.exchanges)} exchanges for {for_date}")

        from skills.conversation.skill import reset_session
        reset_session()

        return summary

    except Exception as e:
        logger.error(f"[compact] Failed for {for_date}: {e}")
        doc.compact_status = "error"
        await doc.save()
        raise RuntimeError(str(e)) from e
