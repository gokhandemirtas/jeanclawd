"""Thin async Spotify Web API client.

Handles Spotify-specific quirks: 204 No Content responses on playback
control endpoints, device resolution, and search.
"""

from typing import Optional

import aiohttp

_API_BASE = "https://api.spotify.com/v1"


class SpotifyClient:
    def __init__(self, access_token: str):
        self._headers = {"Authorization": f"Bearer {access_token}"}

    async def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.get(f"{_API_BASE}{path}", params=params) as resp:
                resp.raise_for_status()
                if resp.status == 204:
                    return None
                return await resp.json()

    async def _put(self, path: str, json: Optional[dict] = None, params: Optional[dict] = None) -> None:
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.put(f"{_API_BASE}{path}", json=json, params=params) as resp:
                resp.raise_for_status()

    async def _post(self, path: str, json: Optional[dict] = None, params: Optional[dict] = None) -> None:
        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.post(f"{_API_BASE}{path}", json=json, params=params) as resp:
                resp.raise_for_status()

    async def get_active_device_id(self) -> Optional[str]:
        data = await self._get("/me/player/devices")
        if not data:
            return None
        devices = data.get("devices", [])
        active = next((d for d in devices if d["is_active"]), None)
        if active:
            return active["id"]
        return devices[0]["id"] if devices else None

    async def now_playing(self) -> Optional[dict]:
        """Returns {"name": ..., "artist": ...} or None if nothing playing."""
        data = await self._get("/me/player")
        if not data or not data.get("item"):
            return None
        item = data["item"]
        return {
            "name": item["name"],
            "artist": ", ".join(a["name"] for a in item["artists"]),
        }

    async def search_track(self, query: str) -> Optional[dict]:
        """Returns {"name": ..., "artist": ..., "uri": ...} for the top result."""
        data = await self._get("/search", params={"q": query, "type": "track", "limit": 1})
        tracks = (data or {}).get("tracks", {}).get("items", [])
        if not tracks:
            return None
        t = tracks[0]
        return {
            "name": t["name"],
            "artist": t["artists"][0]["name"],
            "uri": t["uri"],
        }

    async def play(self, device_id: Optional[str], uris: list[str]) -> None:
        params = {"device_id": device_id} if device_id else None
        await self._put("/me/player/play", json={"uris": uris}, params=params)

    async def queue(self, uri: str, device_id: Optional[str]) -> None:
        params: dict = {"uri": uri}
        if device_id:
            params["device_id"] = device_id
        await self._post("/me/player/queue", params=params)

    async def pause(self, device_id: Optional[str]) -> None:
        params = {"device_id": device_id} if device_id else None
        await self._put("/me/player/pause", params=params)

    async def resume(self, device_id: Optional[str]) -> None:
        params = {"device_id": device_id} if device_id else None
        await self._put("/me/player/play", params=params)

    async def skip(self, device_id: Optional[str]) -> None:
        params = {"device_id": device_id} if device_id else None
        await self._post("/me/player/next", params=params)

    async def previous(self, device_id: Optional[str]) -> None:
        params = {"device_id": device_id} if device_id else None
        await self._post("/me/player/previous", params=params)

    async def set_volume(self, level: int, device_id: Optional[str]) -> None:
        params: dict = {"volume_percent": level}
        if device_id:
            params["device_id"] = device_id
        await self._put("/me/player/volume", params=params)
