"""Generates the OpenAPI 3.0 spec at runtime from the skills registry and API models."""

from api.models import (
	ErrorResponse,
	NoteModel,
	NotesResponse,
	QueryRequest,
	QueryResponse,
	ReminderModel,
	RemindersResponse,
	VoiceResponse,
)


def build_spec() -> dict:
	from skills.registry import SKILLS

	# Derive intent enum and skill descriptions directly from the registry
	intent_enum = [s.id for s in SKILLS]
	skills_table = "\n".join(f"- **{s.id}**: {s.description}" for s in SKILLS)

	# Inline schemas from Pydantic models
	schemas = {
		"QueryRequest": QueryRequest.model_json_schema(),
		"QueryResponse": QueryResponse.model_json_schema(),
		"VoiceResponse": VoiceResponse.model_json_schema(),
		"NoteModel": NoteModel.model_json_schema(),
		"NotesResponse": NotesResponse.model_json_schema(),
		"ReminderModel": ReminderModel.model_json_schema(),
		"RemindersResponse": RemindersResponse.model_json_schema(),
		"ErrorResponse": ErrorResponse.model_json_schema(),
		"Intent": {
			"type": "string",
			"enum": intent_enum,
			"description": "Classified intent. Derived from the skills registry at runtime.",
		},
	}

	return {
		"openapi": "3.0.0",
		"info": {
			"title": "jeanclawd API",
			"version": "1.0.0",
			"description": (
				"Personal AI assistant API. All intelligence runs server-side.\n\n"
				"## Available Skills\n\n"
				f"{skills_table}\n\n"
				"## Authentication\n\n"
				"No application-level auth. Access is restricted to devices on the same "
				"Tailscale network (WireGuard). Ensure Tailscale is connected before making requests."
			),
		},
		"paths": {
			"/query": {
				"post": {
					"summary": "Send a text query",
					"description": (
						"Runs the query through the intent classifier and returns the skill response. "
						f"Valid intents: {', '.join(intent_enum)}"
					),
					"requestBody": {
						"required": True,
						"content": {
							"application/json": {
								"schema": {"$ref": "#/components/schemas/QueryRequest"},
								"example": {"query": "What's the weather like today?"},
							}
						},
					},
					"responses": {
						"200": {
							"description": "Successful response",
							"content": {
								"application/json": {
									"schema": {"$ref": "#/components/schemas/QueryResponse"},
									"example": {"response": "It's 14°C and overcast in London."},
								}
							},
						},
						"400": _error_response("query field is missing or empty"),
					},
				}
			},
			"/voice": {
				"post": {
					"summary": "Send a voice recording",
					"description": (
						"Accepts a recorded audio file (m4a). "
						"Server transcribes via Whisper, classifies intent, generates response, "
						"synthesises speech via pocket-tts, and returns both text and base64 audio."
					),
					"requestBody": {
						"required": True,
						"content": {
							"multipart/form-data": {
								"schema": {
									"type": "object",
									"required": ["audio"],
									"properties": {
										"audio": {
											"type": "string",
											"format": "binary",
											"description": "Recorded audio file (m4a or wav)",
										}
									},
								}
							}
						},
					},
					"responses": {
						"200": {
							"description": "Transcription + spoken response",
							"content": {
								"application/json": {
									"schema": {"$ref": "#/components/schemas/VoiceResponse"},
									"example": {
										"text": "It's 14°C and overcast in London.",
										"audio": "<base64-encoded WAV>",
									},
								}
							},
						},
						"400": _error_response("audio field is missing"),
					},
				}
			},
			"/notes": {
				"get": {
					"summary": "List all notes",
					"description": "Returns all saved notes, newest first.",
					"responses": {
						"200": {
							"description": "List of notes",
							"content": {
								"application/json": {
									"schema": {"$ref": "#/components/schemas/NotesResponse"}
								}
							},
						}
					},
				}
			},
			"/reminders": {
				"get": {
					"summary": "List reminders",
					"description": "Returns reminders sorted by datetime ascending. Excludes completed by default.",
					"parameters": [
						{
							"name": "include_completed",
							"in": "query",
							"required": False,
							"schema": {"type": "boolean", "default": False},
							"description": "Include completed reminders",
						}
					],
					"responses": {
						"200": {
							"description": "List of reminders",
							"content": {
								"application/json": {
									"schema": {"$ref": "#/components/schemas/RemindersResponse"}
								}
							},
						}
					},
				}
			},
		},
		"components": {"schemas": schemas},
	}


def _error_response(example: str) -> dict:
	return {
		"description": "Bad request",
		"content": {
			"application/json": {
				"schema": {"$ref": "#/components/schemas/ErrorResponse"},
				"example": {"error": example},
			}
		},
	}
