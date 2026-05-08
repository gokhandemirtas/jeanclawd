"""Request and response models for the jeanclawd API."""

from typing import Optional

from pydantic import BaseModel, Field

from db.schemas import NoteCategory


# --- /query ---


class QueryRequest(BaseModel):
	query: str = Field(
		..., description="Natural language query sent to the intent classifier"
	)


class QueryResponse(BaseModel):
	response: str = Field(
		..., description="Natural language response from the matched skill"
	)


# --- /tts/voices ---


class VoicesResponse(BaseModel):
	voices: list[str] = Field(..., description="Available TTS voice IDs")
	default: str = Field(..., description="Default voice ID used when none is specified")


# --- /voice ---


class VoiceResponse(BaseModel):
	transcription: str = Field(
		..., description="Whisper transcription of the user's audio input"
	)
	text: str = Field(..., description="Natural language response from the matched skill")
	audio: str = Field(..., description="Base64-encoded WAV audio of the spoken response")


# --- /notes ---


class NoteModel(BaseModel):
	id: str
	title: str = Field(..., description="Plain text title")
	content: str = Field(..., description="Note body in markdown")
	category: Optional[NoteCategory] = Field(
		None, description="read | listen | watch | eat | visit"
	)
	created_at: str = Field(..., description="ISO 8601 datetime")


class NotesResponse(BaseModel):
	notes: list[NoteModel]


# --- /reminders ---


class ReminderModel(BaseModel):
	id: str
	description: str
	reminder_datetime: str = Field(..., description="ISO 8601 datetime")
	is_completed: bool


class RemindersResponse(BaseModel):
	reminders: list[ReminderModel]


# --- errors ---


class ErrorResponse(BaseModel):
	error: str
