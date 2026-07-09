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
- PLAY: play a song, artist, or album
- PLAY_PLAYLIST: play one of the user's saved playlists by name
- PAUSE: pause playback
- STOP: pause playback (alias for PAUSE — Spotify has no separate stop endpoint)
- RESUME: resume playback
- SKIP: skip to the next track
- NEXT: skip to the next track
- PREVIOUS: go back to the previous track
- VOLUME: change the volume (extract integer 0–100)
- NOW_PLAYING: show what is currently playing
- QUEUE: add a song to the queue
- LIST_DEVICES: list available/connected Spotify devices
- LIST_PLAYLISTS: list the user's saved playlists

For PLAY and QUEUE involving a single song, set `target` to the search query (e.g. "Bohemian Rhapsody Queen").
For PLAY_PLAYLIST, set `target` to the playlist name as the user said it.
If the user refers to a list of songs mentioned earlier in the conversation (e.g. "play them all"), extract each song title + artist into `targets` as separate strings.
If the user says "first song", "second song", etc., resolve that to the specific song from the conversation context and set `target`.
For VOLUME, extract the level as an integer 0–100.
If the user specifies a device (e.g. "on my Mac", "on the TV", "on iPhone"), extract the device name.
Leave optional fields as null if not applicable.
"""


class _SpotifyAction(BaseModel):
	action: Literal[
		"PLAY",
		"PLAY_PLAYLIST",
		"PAUSE",
		"STOP",
		"RESUME",
		"SKIP",
		"NEXT",
		"PREVIOUS",
		"NOW_PLAYING",
		"QUEUE",
		"LIST_DEVICES",
		"LIST_PLAYLISTS",
	]
	target: Optional[str] = Field(
		None, description="Search query for PLAY/QUEUE, or playlist name for PLAY_PLAYLIST"
	)
	targets: Optional[list[str]] = Field(
		None,
		description="List of search queries when playing multiple songs",
	)
	volume: Optional[int] = Field(
		None, description="Volume 0-100 for VOLUME action", ge=0, le=100
	)
	device: Optional[str] = Field(
		None, description="Device name to play on, if specified by the user"
	)


def _resolve_device(hint: str, devices: list[dict]) -> dict | None:
    if not devices:
        return None
    from rank_bm25 import BM25Okapi

    corpus = [d["name"].lower().split() for d in devices]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(hint.lower().split())
    best_idx = int(scores.argmax())
    return devices[best_idx] if scores[best_idx] > 0 else None


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

	try:
		with Live(
			Spinner("dots", text=Text("Spotifying...", style="dim")),
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
	except Exception as e:
		logger.error(f"Spotify action classification failed: {e}")
		return "I couldn't understand that as a Spotify command. Try 'play [song]', 'pause', 'what's playing?', or 'next'."

	logger.debug(f"Spotify action: {action}")

	try:
		token = await _auth.get_token()
		client = SpotifyClient(token)

		if action.device:
			devices = await client.get_devices()
			match = _resolve_device(action.device, devices)
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

		if action.action == "LIST_PLAYLISTS":
			playlists = await client.get_user_playlists()
			if not playlists:
				return "No playlists found in your Spotify library."
			lines = [f"- {p['name']} ({p['track_count']} tracks)" for p in playlists]
			return "Your Spotify playlists:\n" + "\n".join(lines)

		if action.action == "PLAY_PLAYLIST":
			name_hint = action.target or query
			playlists = await client.get_user_playlists()
			if not playlists:
				return "No playlists found in your Spotify library."
			from rank_bm25 import BM25Okapi
			corpus = [p["name"].lower().split() for p in playlists]
			bm25 = BM25Okapi(corpus)
			scores = bm25.get_scores(name_hint.lower().split())
			best_idx = int(scores.argmax())
			match = playlists[best_idx] if scores[best_idx] > 0 else playlists[0]
			await client.play_playlist(device_id, match["uri"])
			import asyncio

			await asyncio.sleep(0.4)
			state = await client.now_playing()
			if not state or not state.get("is_playing"):
				return f"Couldn't start playlist '{match['name']}' — check that a device is active and try again."
			return f"Playing playlist: {match['name']}."

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
			import asyncio

			await asyncio.sleep(0.5)
			state = await client.now_playing()
			if not state or not state.get("is_playing"):
				return "Couldn't start playback on Spotify — check that a device is active and try again."
			if state.get("uri") and state["uri"] != tracks[0]["uri"]:
				return f"Playback started, but '{state['name']}' is playing instead of '{tracks[0]['name']}'."
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

		if action.action in ("PAUSE", "STOP"):
			await client.pause(device_id)
			import asyncio

			await asyncio.sleep(0.4)
			state = await client.now_playing()
			if state and state.get("is_playing"):
				return "Couldn't stop Spotify — playback is still running on the active device."
			return "Stopped Spotify." if action.action == "STOP" else "Paused Spotify."

		if action.action == "RESUME":
			await client.resume(device_id)
			import asyncio

			await asyncio.sleep(0.4)
			state = await client.now_playing()
			if not state or not state.get("is_playing"):
				return "Couldn't resume Spotify — check that a device is active and try again."
			return f"Resumed Spotify. Now playing: {state['name']} by {state['artist']}."

		if action.action in ("SKIP", "NEXT"):
			await client.skip(device_id)
			import asyncio
			await asyncio.sleep(0.6)
			track = await client.now_playing()
			if not track:
				return "Couldn't skip — no track is playing or the device is inactive."
			return f"Skipped to: {track['name']} by {track['artist']}."

		if action.action == "PREVIOUS":
			await client.previous(device_id)
			import asyncio
			await asyncio.sleep(0.6)
			track = await client.now_playing()
			if not track:
				return "Couldn't go back — no track is playing or the device is inactive."
			return f"Now playing: {track['name']} by {track['artist']}."

		if action.action == "VOLUME":
			level = action.volume if action.volume is not None else 50
			await client.set_volume(level, device_id)
			import asyncio
			await asyncio.sleep(0.3)
			current = await client.get_volume()
			if current is None:
				return f"Couldn't change volume — no active Spotify device. Tried to set {level}%."
			if current != level:
				return f"Tried to set volume to {level}% but Spotify reports {current}% — something went wrong."
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
