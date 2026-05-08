"""Sub-classification prompt for the Notes skill."""

NOTES_INTENT_PROMPT = """Classify the user's notes request into exactly one sub-intent.

## Sub-intents

### CREATE
Save a new note.
- "save a note about...", "note that...", "remember that...", "add a note"

### LIST
List existing notes, optionally filtered by category.
- "show my notes", "list notes", "what notes do I have", "show my read notes", "notes tagged watch"

### READ
Read the content of a specific note by title.
- "read my note about X", "open the note called X", "show me the X note"

### DELETE
Delete a specific note by title.
- "delete the note about X", "remove my note on X"

## Response format
Respond with EXACTLY ONE of: CREATE, LIST, READ, DELETE
"""
