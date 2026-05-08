"""Create a new note."""

import os
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field
from utils.log import logger

from db.schemas import Note, NoteCategory
from utils.llm_structured_output import generate_structured_output


class _NoteExtraction(BaseModel):
    title: str = Field(description="Plain text title for the note")
    content: str = Field(description="Note body in markdown. Capture all relevant details.")
    category: Optional[NoteCategory] = Field(
        default=None,
        description="Category tag: 'read' (books/articles), 'listen' (music/podcasts), "
                    "'watch' (films/shows), 'eat' (food/restaurants), 'visit' (places). "
                    "Omit if none clearly applies.",
    )


_EXTRACT_PROMPT = """Extract a structured note from the user's request.

Title: short, plain text (no markdown).
Content: record ONLY what the user explicitly stated — do not add, infer, or invent any details.
Category: only set if clearly one of: read, listen, watch, eat, visit.
"""


async def handle_create_note(query: str) -> str:
    extracted = generate_structured_output(
        model_name=os.environ["MODEL_GENERALIST"],
        user_prompt=query,
        system_prompt=_EXTRACT_PROMPT,
        pydantic_model=_NoteExtraction,
        max_tokens=512,
    )

    note = Note(
        created_at=datetime.now(),
        title=extracted.title,
        content=extracted.content,
        category=extracted.category,
    )
    await note.insert()

    tag = f" [{extracted.category}]" if extracted.category else ""
    logger.debug(f"[notes] created '{extracted.title}'{tag}")
    return f"Saved: **{extracted.title}**{tag}\n\n{extracted.content}"


async def handle_create_note_from_context(title: str, content: str, category: Optional[NoteCategory] = None) -> str:
    """Create a note directly from pre-composed title/content (used by /note command)."""
    note = Note(
        created_at=datetime.now(),
        title=title,
        content=content,
        category=category,
    )
    await note.insert()

    tag = f" [{category}]" if category else ""
    logger.debug(f"[notes] created from context '{title}'{tag}")
    return f"Saved: **{title}**{tag}"
