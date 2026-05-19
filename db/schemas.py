from datetime import date, datetime
from typing import Literal, Optional

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


class ConversationExchange(BaseModel):
	timestamp: datetime
	question: str
	answer: str


class Conversations(Document):
	date: date
	exchanges: list[ConversationExchange] = []
	compacted: Optional[str] = None
	compact_status: Literal["idle", "pending", "complete", "error"] = "idle"


class Journal(Document):
	datestamp: date
	summary: str

	class Settings:
		name = "Journal"


class Note(Document):
	created_at: datetime
	title: str  # plain text
	content: str  # markdown
	category: Optional[NoteCategory] = None

	class Settings:
		name = "Notes"


class VoiceSettings(Document):
	active_voice_id: str = "azelma"


class Preferences(BaseModel):
	agentPersona: str = "jean"
	agentVoiceId: str = "azelma"
	favSpotifyDevice: Optional[str] = None


class UserPersona(Document):
	last_updated: datetime
	preferences: Preferences = Field(default_factory=Preferences)

	# Tier 1 — ask freely
	name: Optional[str] = Field(
		default=None,
		json_schema_extra={"tier": 1, "question": "what's your name, by the way?"},
	)
	location: Optional[str] = Field(
		default=None,
		json_schema_extra={"tier": 1, "question": "whereabouts are you based?"},
	)
	country: Optional[str] = Field(
		default=None, json_schema_extra={"tier": 1, "question": "which country are you in?"}
	)
	occupation: Optional[str] = Field(
		default=None, json_schema_extra={"tier": 1, "question": "what do you do for work?"}
	)
	diet: Optional[str] = Field(
		default=None,
		json_schema_extra={"tier": 1, "question": "do you follow any particular diet?"},
	)
	interests: list[str] = Field(
		default_factory=list,
		json_schema_extra={
			"tier": 1,
			"question": "what are your main hobbies or interests?",
		},
	)
	music: list[str] = Field(
		default_factory=list,
		json_schema_extra={
			"tier": 1,
			"question": "what type of music tickles your ears?",
		},
	)

	# Tier 2 — ask after some rapport
	age: Optional[int] = Field(
		default=None,
		json_schema_extra={
			"tier": 2,
			"question": "roughly how old are you, if you don't mind me asking?",
		},
	)
	gender: Optional[str] = Field(
		default=None,
		json_schema_extra={"tier": 2, "question": "how do you identify, gender-wise?"},
	)
	relationship_status: Optional[str] = Field(
		default=None,
		json_schema_extra={
			"tier": 2,
			"question": "are you in a relationship or living solo these days?",
		},
	)
	has_children: Optional[bool] = Field(
		default=None, json_schema_extra={"tier": 2, "question": "do you have any kids?"}
	)
	languages: list[str] = Field(
		default_factory=list,
		json_schema_extra={"tier": 2, "question": "what languages do you speak?"},
	)
	nationality: Optional[str] = Field(
		default=None, json_schema_extra={"tier": 2, "question": "what's your nationality?"}
	)

	# Tier 3 — ask only when naturally relevant
	political_leaning: Optional[str] = Field(
		default=None,
		json_schema_extra={
			"tier": 3,
			"question": "how would you describe your political views, if you're comfortable sharing?",
		},
	)
	religion: Optional[str] = Field(
		default=None,
		json_schema_extra={
			"tier": 3,
			"question": "do you follow any religion or spiritual practice?",
		},
	)
	education: Optional[str] = Field(
		default=None,
		json_schema_extra={"tier": 3, "question": "what's your educational background?"},
	)

	# Tier 0 — inferred from conversation, never directly asked
	communication_style: Optional[str] = Field(
		default=None, json_schema_extra={"tier": 0}
	)
	character_traits: list[str] = Field(
		default_factory=list, json_schema_extra={"tier": 0}
	)
	likes: list[str] = Field(default_factory=list, json_schema_extra={"tier": 0})
	dislikes: list[str] = Field(default_factory=list, json_schema_extra={"tier": 0})
