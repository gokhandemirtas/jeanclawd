import os
from typing import Literal, Optional

import aiohttp
from pydantic import BaseModel, Field
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from skills.spotify.spotify_client import SpotifyClient
from utils.auth.spotify import SpotifyAuthProvider
from utils.llm_structured_output import generate_structured_output
from utils.log import logger

_console = Console(stderr=True, force_terminal=True)
_auth = SpotifyAuthProvider()

_SPOTIFY_PROMPT = """You are a Spotify command parser. Extract the action, target(s), and device from the user's request.

Actions:
- PLAY: play a song, artist, album, or playlist
- PAUSE: pause playback
- RESUME: resume playback
- SKIP: skip to the next track
- NEXT: skip to the next track
- PREVIOUS: go back to the previous track
- VOLUME: change the volume (extract integer 0–100)
- NOW_PLAYING: show what is currently playing
- QUEUE: add a song to the queue
- LIST_DEVICES: list available/connected Spotify devices

For PLAY and QUEUE involving a single song, set `target` to the search query (e.g. "Bohemian Rhapsody Queen").
If the user refers to a list of songs mentioned earlier in the conversation (e.g. "play them all", "play the playlist"), extract each song title + artist into `targets` as separate strings.
If the user says "first song", "second song", etc., resolve that to the specific song from the conversation context and set `target`.
For VOLUME, extract the level as an integer 0–100.
If the user specifies a device (e.g. "on my Mac", "on the TV", "on iPhone"), extract the device name.
Leave optional fields as null if not applicable.
"""


class _SpotifyAction(BaseModel):
	action: Literal[
		"PLAY",
		"PAUSE",
		"RESUME",
		"SKIP",
		"NEXT",
		"PREVIOUS",
		"NOW_PLAYING",
		"QUEUE",
		"LIST_DEVICES",
	]
	target: Optional[str] = Field(
		None, description="Search query for a single PLAY/QUEUE action"
	)
	targets: Optional[list[str]] = Field(
		None,
		description="List of search queries when playing multiple songs (e.g. a playlist)",
	)
	volume: Optional[int] = Field(
		None, description="Volume 0-100 for VOLUME action", ge=0, le=100
	)
	device: Optional[str] = Field(
		None, description="Device name to play on, if specified by the user"
	)


def _recent_history(n_messages: int = 8) -> str:
	from skills.conversation.skill import _history

	messages = list(_history)[-n_messages:]
	if not messages:
		return ""
	lines = []
	for msg in messages:
		role = "User" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
		lines.append(f"{role}: {msg.content}")
	return "\n".join(lines)


async def handle_spotify(query: str) -> str:
	context = _recent_history()
	enriched = (
		f"Recent conversation:\n{context}\n\nUser's request: {query}" if context else query
	)

	with Live(
		Spinner("dots", text=Text("thinking...", style="dim")),
		console=_console,
		transient=True,
	):
		action = generate_structured_output(
			model_name=os.environ["MODEL_GENERALIST"],
			user_prompt=enriched,
			system_prompt=_SPOTIFY_PROMPT,
			pydantic_model=_SpotifyAction,
			max_tokens=300,
		)

	logger.debug(f"Spotify action: {action}")

	try:
		token = await _auth.get_token()
		client = SpotifyClient(token)

		if action.device:
			devices = await client.get_devices()
			needle = action.device.lower()
			match = next((d for d in devices if needle in d["name"].lower()), None)
			if match:
				device_id = match["id"]
			else:
				names = ", ".join(d["name"] for d in devices) or "none found"
				return f"Couldn't find a device matching '{action.device}'. Available: {names}."
		else:
			device_id = await client.get_active_device_id()

		if action.action == "LIST_DEVICES":
			devices = await client.get_devices()
			if not devices:
				return "No Spotify devices found. Open Spotify on at least one device."
			lines = []
			for d in devices:
				status = "active" if d["is_active"] else "inactive"
				lines.append(f"- {d['name']} ({d['type']}, {status})")
			return "Connected Spotify devices:\n" + "\n".join(lines)

		if action.action == "NOW_PLAYING":
			track = await client.now_playing()
			if not track:
				return "Nothing is currently playing on Spotify."
			return f"Now playing: {track['name']} by {track['artist']}."

		if action.action == "PLAY":
			queries = action.targets or ([action.target] if action.target else [query])
			tracks = [t for q in queries if (t := await client.search_track(q))]
			if not tracks:
				return f"Couldn't find anything on Spotify for: {', '.join(queries)}"
			await client.play(device_id, [t["uri"] for t in tracks])
			if len(tracks) == 1:
				return f"Playing {tracks[0]['name']} by {tracks[0]['artist']}."
			return f"Playing {len(tracks)} tracks, starting with {tracks[0]['name']} by {tracks[0]['artist']}."

		if action.action == "QUEUE":
			target = action.target or query
			track = await client.search_track(target)
			if not track:
				return f"Couldn't find anything on Spotify for: {target}"
			await client.queue(track["uri"], device_id)
			return f"Added {track['name']} by {track['artist']} to the queue."

		if action.action == "PAUSE":
			await client.pause(device_id)
			return "Paused Spotify."

		if action.action == "RESUME":
			await client.resume(device_id)
			return "Resumed Spotify."

		if action.action in ("SKIP", "NEXT"):
			await client.skip(device_id)
			return "Skipped to the next track."

		if action.action == "PREVIOUS":
			await client.previous(device_id)
			return "Going back to the previous track."

		if action.action == "VOLUME":
			level = action.volume if action.volume is not None else 50
			await client.set_volume(level, device_id)
			return f"Volume set to {level}%."

	except aiohttp.ClientResponseError as e:
		logger.error(f"Spotify API error {e.status}: {e.message}")
		if e.status == 404:
			return "No active Spotify device found. Open Spotify on a device and try again."
		if e.status == 401:
			return "Spotify token expired. This should auto-refresh — please try again."
		return "Spotify returned an error. Make sure a device is active and try again."
	except Exception as e:
		logger.error(f"Spotify skill error: {e}")
		return "Something went wrong with Spotify."

	return "I didn't understand that Spotify command."
