"""List notes, optionally filtered by category."""

import os
from typing import Optional

from pydantic import BaseModel, Field
from utils.log import logger

from db.schemas import Note, NoteCategory
from utils.llm_structured_output import generate_structured_output


class _ListFilter(BaseModel):
    category: Optional[NoteCategory] = Field(
        default=None,
        description="Filter by category if mentioned: read, listen, watch, eat, visit. Omit for all notes.",
    )


_FILTER_PROMPT = "Extract the category filter from the user's list request, if any. Valid values: read, listen, watch, eat, visit."

_CATEGORY_KEYWORDS = {"read", "listen", "watch", "eat", "visit"}


async def handle_list_notes(query: str) -> str:
    category = None
    query_words = set(query.lower().split())
    if query_words & _CATEGORY_KEYWORDS:
        try:
            extracted = generate_structured_output(
                model_name=os.environ["MODEL_GENERALIST"],
                user_prompt=query,
                system_prompt=_FILTER_PROMPT,
                pydantic_model=_ListFilter,
                max_tokens=50,
            )
            category = extracted.category
        except Exception:
            logger.info("[notes] category extraction failed, listing all notes")

    logger.info(f"[notes] list: extracted category_filter={category!r}")

    filter_query = Note.find()
    if category:
        filter_query = Note.find(Note.category == category)

    notes = await filter_query.sort(-Note.created_at).to_list()
    logger.info(f"[notes] list: category_filter={category!r}, found={len(notes)}")

    if not notes:
        label = f"[{category}] " if category else ""
        return f"No {label} notes found."

    lines = []
    for n in notes:
        tag = f" [{n.category}]" if n.category else ""
        date_str = n.created_at.strftime("%Y-%m-%d")
        lines.append(f"- **{n.title}**{tag} — {date_str}")

    header = f"Notes [{category}]:" if category else "All notes:"
    logger.debug(f"[notes] listed {len(notes)} notes")
    return f"{header}\n" + "\n".join(lines)
