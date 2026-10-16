from datetime import date, datetime
from typing import Literal
from uuid import UUID

from beanie import Document
from pydantic import BaseModel, Field

NoteCategory = Literal["read", "listen", "watch", "eat", "visit"]


class HueLight(BaseModel):
	id: int
	name: str
	is_on: bool
	bri: int | None = None
	hue: int | None = None
	sat: int | None = None


class HueLightGroup(BaseModel):
	id: int
	name: str


class Weather(Document):
	datestamp: date
	location: str
	current_temperature: int
	forecast: list[str]


class Credentials(Document):
	hue_username: str


class ServiceCredentials(Document):
	"""Generic per-service credential store. One document per service (e.g. 'spotify', 'hue')."""

	service: str
	data: dict  # flexible payload — tokens, usernames, expiry timestamps, etc.


class HueConfiguration(Document):
	groups: list[HueLightGroup]
	lights: list[HueLight]
	last_updated: date


class Newsfeed(Document):
	datestamp: date
	content: str  # Large text field for storing full RSS feed content


class Reminders(Document):
	reminder_datetime: datetime  # Main datetime for the reminder
	reminder_end_time: datetime | None = None  # Optional end time for time ranges
	description: str
	created_at: date
	is_completed: bool = False


class Plans(Document):
	created_at: datetime
	ask: str
	plan: str


class Session(Document):
	"""A resumable conversation thread."""

	session_id: UUID
	user_id: str = "default"
	title: str | None = None
	summary: str | None = None
	current_topic: str | None = None
	last_research_report: str | None = None
	created_at: datetime
	last_active_at: datetime
	status: Literal["active", "closed"] = "active"

	class Settings:
		name = "Sessions"


class Interaction(Document):
	"""Immutable record of one completed user/assistant turn."""

	user_id: str = "default"
	session_id: str = "default"
	created_at: datetime = Field(default_factory=datetime.now)
	user_message: str
	assistant_message: str
	intent: str | None = None

	class Settings:
		name = "Interactions"


class Note(Document):
	created_at: datetime
	title: str  # plain text
	content: str  # markdown
	category: NoteCategory | None = None

	class Settings:
		name = "Notes"


class UserSettings(Document):
	"""Explicit application settings, deliberately separate from learned memory."""

	user_id: str = "default"
	agent_persona: str = "jean"
	agent_voice_id: str = "thomas"
	fav_spotify_device: str | None = None
	news_length: Literal["short", "medium", "long"] = "medium"

	class Settings:
		name = "UserSettings"


class ResearchDocument(Document):
	"""Metadata for one completed deep-research report."""

	research_id: str
	user_id: str = "default"
	topic: str
	original_query: str
	report_path: str
	generated_at: datetime
	content_hash: str
	source_urls: list[str] = Field(default_factory=list)

	class Settings:
		name = "ResearchDocuments"
		indexes = [
			[("user_id", 1), ("generated_at", -1)],
			[("user_id", 1), ("topic", 1)],
			[("report_path", 1), ("content_hash", 1)],
		]


class ResearchChunk(Document):
	"""Independently retrievable section of a research report."""

	research_id: str
	user_id: str = "default"
	chunk_id: str
	section: str
	content: str
	source_markers: list[str] = Field(default_factory=list)
	embedding: list[float] | None = None

	class Settings:
		name = "ResearchChunks"
		indexes = [
			[("user_id", 1), ("research_id", 1)],
			[("user_id", 1), ("section", 1)],
		]


class IntrospectionEmbedding(Document):
	"""Cached embedding for one introspection reflection file."""

	filename: str
	content_hash: str
	embedding: list[float]

	class Settings:
		name = "IntrospectionEmbeddings"
		indexes = [
			[("filename", 1)],
		]
