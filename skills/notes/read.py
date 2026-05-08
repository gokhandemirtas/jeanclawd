"""Read a specific note by title."""

import re

from utils.log import logger

from db.schemas import Note


async def handle_read_note(query: str) -> str:
    # Extract a search term — strip common command phrases
    search = re.sub(
        r"(read|open|show|get|find|display)\s+(me\s+)?(the\s+)?(note\s+(about|called|titled|on)\s+)?",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip().strip('"').strip("'")

    if not search:
        return "Please specify which note you'd like to read."

    # Case-insensitive partial title match
    notes = await Note.find(
        {"title": {"$regex": search, "$options": "i"}}
    ).sort(-Note.created_at).to_list()

    if not notes:
        return f"No note found matching \"{search}\"."

    if len(notes) > 1:
        matches = ", ".join(f'"{n.title}"' for n in notes[:5])
        return f"Multiple notes match \"{search}\": {matches}. Please be more specific."

    note = notes[0]
    tag = f" [{note.category}]" if note.category else ""
    date_str = note.created_at.strftime("%Y-%m-%d %H:%M")
    logger.debug(f"[notes] read '{note.title}'")
    return f"# {note.title}{tag}\n*{date_str}*\n\n{note.content}"
