"""Delete a note by title."""

import re

from utils.log import logger

from db.schemas import Note


async def handle_delete_note(query: str) -> str:
    search = re.sub(
        r"(delete|remove|erase)\s+(the\s+)?(note\s+(about|called|titled|on)\s+)?",
        "",
        query,
        flags=re.IGNORECASE,
    ).strip().strip('"').strip("'")

    if not search:
        return "Please specify which note you'd like to delete."

    notes = await Note.find(
        {"title": {"$regex": search, "$options": "i"}}
    ).to_list()

    if not notes:
        return f"No note found matching \"{search}\"."

    if len(notes) > 1:
        matches = ", ".join(f'"{n.title}"' for n in notes[:5])
        return f"Multiple notes match \"{search}\": {matches}. Please be more specific."

    note = notes[0]
    await note.delete()
    logger.debug(f"[notes] deleted '{note.title}'")
    return f"Deleted note: **{note.title}**"
