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

_SPOTIFY_PROMPT = """You are a Spotify command parser. Extract the action and target from the user's query.

Actions:
- PLAY: play a song, artist, album, or playlist
- PAUSE: pause playback
- RESUME: resume playback
- SKIP: skip to the next track
- PREVIOUS: go back to the previous track
- VOLUME: change the volume (extract integer 0–100)
- NOW_PLAYING: show what is currently playing
- QUEUE: add a song to the queue

For PLAY and QUEUE, extract the search query (e.g. "Radiohead", "Bohemian Rhapsody").
For VOLUME, extract the level as an integer 0–100.
Leave optional fields as null if not applicable.
"""


class _SpotifyAction(BaseModel):
    action: Literal["PLAY", "PAUSE", "RESUME", "SKIP", "PREVIOUS", "VOLUME", "NOW_PLAYING", "QUEUE"]
    target: Optional[str] = Field(None, description="Search query for PLAY/QUEUE actions")
    volume: Optional[int] = Field(None, description="Volume 0-100 for VOLUME action", ge=0, le=100)


async def handle_spotify(query: str) -> str:
    with Live(Spinner("dots", text=Text("thinking...", style="dim")), console=_console, transient=True):
        action = generate_structured_output(
            model_name=os.environ["MODEL_GENERALIST"],
            user_prompt=query,
            system_prompt=_SPOTIFY_PROMPT,
            pydantic_model=_SpotifyAction,
            max_tokens=100,
        )

    logger.debug(f"Spotify action: {action}")

    try:
        token = await _auth.get_token()
        client = SpotifyClient(token)
        device_id = await client.get_active_device_id()

        if action.action == "NOW_PLAYING":
            track = await client.now_playing()
            if not track:
                return "Nothing is currently playing on Spotify."
            return f"Now playing: {track['name']} by {track['artist']}."

        if action.action == "PLAY":
            target = action.target or query
            track = await client.search_track(target)
            if not track:
                return f"Couldn't find anything on Spotify for: {target}"
            await client.play(device_id, [track["uri"]])
            return f"Playing {track['name']} by {track['artist']}."

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

        if action.action == "SKIP":
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
