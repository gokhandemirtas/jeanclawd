from skills.base import Skill
from skills.notes.skill import handle_notes

notes_skill = Skill(
    id="NOTES",
    description="Create, list, read, and delete personal notes with optional category tags (read, listen, watch, eat, visit)",
    system_prompt=None,
    handle=handle_notes,
    needs_wrapping=False,
)
