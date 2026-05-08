"""Notes skill — create, list, read, and delete notes."""

import os
from typing import Literal

from pydantic import BaseModel

from skills.notes.create import handle_create_note
from skills.notes.delete import handle_delete_note
from skills.notes.list import handle_list_notes
from skills.notes.prompts import NOTES_INTENT_PROMPT
from skills.notes.read import handle_read_note
from utils.llm_structured_output import generate_structured_output
from utils.log import logger


class _NotesIntentResponse(BaseModel):
    sub_intent: Literal["CREATE", "LIST", "READ", "DELETE"]


def _classify_sub_intent(query: str) -> str:
    result = generate_structured_output(
        model_name=os.environ["MODEL_GENERALIST"],
        user_prompt=query,
        system_prompt=NOTES_INTENT_PROMPT,
        pydantic_model=_NotesIntentResponse,
        max_tokens=50,
    )
    return result.sub_intent


async def handle_notes(query: str) -> str:
    sub_intent = _classify_sub_intent(query)
    logger.debug(f"[notes] sub_intent='{sub_intent}'")

    match sub_intent:
        case "CREATE":
            return await handle_create_note(query)
        case "LIST":
            return await handle_list_notes(query)
        case "READ":
            return await handle_read_note(query)
        case "DELETE":
            return await handle_delete_note(query)
        case _:
            return "Could not determine what you'd like to do with your notes."
