"""Spotify OAuth2 Authorization Code flow.

On first use, opens the Spotify auth URL in the default browser and starts a
temporary aiohttp server on localhost:8888 to catch the redirect. Tokens are
stored in ServiceCredentials(service="spotify") and auto-refreshed when expired.
"""

import asyncio
import base64
import os
import time
import webbrowser
from urllib.parse import urlencode

import aiohttp
from aiohttp import web

from db.schemas import ServiceCredentials
from utils.auth.base import AuthProvider
from utils.log import logger

_ACCOUNTS_BASE = "https://accounts.spotify.com"
_SCOPES = " ".join([
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
])
_REDIRECT_URI = "http://127.0.0.1:8888/callback"
_CALLBACK_TIMEOUT = 5000  # seconds to wait for the browser redirect


class SpotifyAuthProvider(AuthProvider):
    service_id = "spotify"
    display_name = "Spotify"

    async def is_connected(self) -> bool:
        creds = await ServiceCredentials.find_one(ServiceCredentials.service == "spotify")
        return creds is not None and "refresh_token" in creds.data

    async def get_token(self) -> str:
        """Return a valid access token, refreshing if within 60 s of expiry."""
        creds = await ServiceCredentials.find_one(ServiceCredentials.service == "spotify")
        if not creds:
            raise RuntimeError("Spotify is not connected. Please try again to start the auth flow.")

        if time.time() >= creds.data.get("expires_at", 0) - 60:
            logger.debug("Spotify access token expired — refreshing")
            new_tokens = await self._refresh_token(creds.data["refresh_token"])
            creds.data.update(new_tokens)
            await creds.save()

        return creds.data["access_token"]

    async def connect(self) -> str:
        client_id = os.environ.get("SPOTIFY_CLIENT_ID")
        if not client_id:
            return "SPOTIFY_CLIENT_ID is not set. Add it to your .env file and restart."

        auth_url = self._build_auth_url(client_id)

        code_future: asyncio.Future = asyncio.get_event_loop().create_future()

        async def _callback(request: web.Request) -> web.Response:
            code = request.rel_url.query.get("code")
            error = request.rel_url.query.get("error")
            if code and not code_future.done():
                code_future.set_result(code)
                return web.Response(text="<html><body><h2>Spotify connected! You can close this tab.</h2></body></html>", content_type="text/html")
            if not code_future.done():
                code_future.set_exception(Exception(error or "auth_denied"))
            return web.Response(text="<html><body><h2>Auth failed. Please try again.</h2></body></html>", content_type="text/html")

        app = web.Application()
        app.router.add_get("/callback", _callback)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "localhost", 8888)

        try:
            await site.start()
        except OSError:
            await runner.cleanup()
            return "Port 8888 is already in use. Free it and try again."

        webbrowser.open(auth_url)
        logger.info(f"Spotify auth URL opened in browser. Waiting for callback (up to {_CALLBACK_TIMEOUT}s)...")

        try:
            code = await asyncio.wait_for(code_future, timeout=_CALLBACK_TIMEOUT)
            tokens = await self._exchange_code(code)
            await self._save_tokens(tokens)
            return f"Spotify connected successfully!"
        except asyncio.TimeoutError:
            return "Spotify connection timed out. Please try again."
        except Exception as e:
            logger.error(f"Spotify auth error: {e}")
            return f"Spotify connection failed: {e}"
        finally:
            await runner.cleanup()

    def _build_auth_url(self, client_id: str) -> str:
        params = urlencode({
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": _REDIRECT_URI,
            "scope": _SCOPES,
        })
        return f"{_ACCOUNTS_BASE}/authorize?{params}"

    def _basic_auth_header(self) -> str:
        client_id = os.environ["SPOTIFY_CLIENT_ID"]
        client_secret = os.environ["SPOTIFY_CLIENT_SECRET"]
        encoded = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        return f"Basic {encoded}"

    async def _exchange_code(self, code: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_ACCOUNTS_BASE}/api/token",
                headers={
                    "Authorization": self._basic_auth_header(),
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": _REDIRECT_URI,
                },
            ) as resp:
                resp.raise_for_status()
                body = await resp.json()

        return {
            "access_token": body["access_token"],
            "refresh_token": body["refresh_token"],
            "expires_at": time.time() + body["expires_in"],
            "scope": body.get("scope", ""),
        }

    async def _refresh_token(self, refresh_token: str) -> dict:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{_ACCOUNTS_BASE}/api/token",
                headers={
                    "Authorization": self._basic_auth_header(),
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            ) as resp:
                resp.raise_for_status()
                body = await resp.json()

        return {
            "access_token": body["access_token"],
            # Spotify may or may not rotate the refresh token
            "refresh_token": body.get("refresh_token", refresh_token),
            "expires_at": time.time() + body["expires_in"],
            "scope": body.get("scope", ""),
        }

    async def _save_tokens(self, tokens: dict) -> None:
        existing = await ServiceCredentials.find_one(ServiceCredentials.service == "spotify")
        if existing:
            existing.data = tokens
            await existing.save()
        else:
            await ServiceCredentials(service="spotify", data=tokens).insert()
