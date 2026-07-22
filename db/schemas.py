from datetime import date, datetime
from typing import Literal, Optional
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
	hueUsername: str


class ServiceCredentials(Document):
	"""Generic per-service credential store. One document per service (e.g. 'spotify', 'hue')."""

	service: str
	data: dict  # flexible payload — tokens, usernames, expiry timestamps, etc.


class HueConfiguration(Document):
	groups: list[HueLightGroup]
	lights: list[HueLight]
	lastUpdated: date


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
	title: Optional[str] = None
	summary: Optional[str] = None
	created_at: datetime
	last_active_at: datetime
	status: Literal["active", "closed"] = "active"

	class Settings:
		name = "Sessions"


class Interaction(Document):
	"""Immutable record of one completed user/assistant turn."""
	user_id: str = "default"
	session_id: str = "default"
	created_at: datetime
	user_message: str
	assistant_message: str
	intent: Optional[str] = None

	class Settings:
		name = "Interactions"


MemoryNamespace = Literal["identity", "preference", "interest", "relationship", "context"]
MemoryStatus = Literal["active", "superseded", "forgotten"]
MemoryWorkStatus = Literal["pending", "processing", "complete", "error"]


class MemoryFact(Document):
	"""A curated, attributable fact used for cross-session recall."""
	user_id: str = "default"
	namespace: MemoryNamespace
	key: str
	value: str
	confidence: float = 0.8
	source_interaction_ids: list[str] = Field(default_factory=list)
	status: MemoryStatus = "active"
	first_observed_at: datetime
	last_confirmed_at: datetime
	expires_at: Optional[datetime] = None

	class Settings:
		name = "MemoryFacts"


class MemoryWorkItem(Document):
	"""Durable outbox item for deferred memory extraction."""
	user_id: str = "default"
	interaction_id: str
	created_at: datetime
	status: MemoryWorkStatus = "pending"
	attempts: int = 0
	error: Optional[str] = None

	class Settings:
		name = "MemoryWorkItems"


class Note(Document):
	created_at: datetime
	title: str  # plain text
	content: str  # markdown
	category: Optional[NoteCategory] = None

	class Settings:
		name = "Notes"


class UserSettings(Document):
	"""Explicit application settings, deliberately separate from learned memory."""
	user_id: str = "default"
	agent_persona: str = "jean"
	agent_voice_id: str = "azelma"
	fav_spotify_device: Optional[str] = None
	news_length: Literal["short", "medium", "long"] = "medium"

	class Settings:
		name = "UserSettings"


class UserProfile(Document):
	"""Small, fast read model for identity details used at startup and in greetings."""
	user_id: str = "default"
	display_name: Optional[str] = None
	birth_date: Optional[str] = None
	gender: Optional[str] = None
	location: Optional[str] = None
	updated_at: datetime = Field(default_factory=datetime.now)

	class Settings:
		name = "UserProfiles"
