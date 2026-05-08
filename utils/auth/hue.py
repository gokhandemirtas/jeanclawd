"""Philips Hue bridge pairing auth provider.

The Hue bridge uses a "press the link button" flow instead of OAuth.
On connect(), the user is prompted to press the button, then the bridge
is polled for the generated username (API key), which is stored in
ServiceCredentials(service="hue").
"""

import asyncio
import os

from db.schemas import ServiceCredentials
from utils.auth.base import AuthProvider
from utils.log import logger


class HueAuthProvider(AuthProvider):
    service_id = "hue"
    display_name = "Philips Hue"

    async def is_connected(self) -> bool:
        creds = await ServiceCredentials.find_one(ServiceCredentials.service == "hue")
        return creds is not None and "username" in creds.data

    async def get_username(self) -> str:
        """Return the stored Hue bridge username (API key)."""
        creds = await ServiceCredentials.find_one(ServiceCredentials.service == "hue")
        if not creds:
            raise RuntimeError("Hue bridge is not connected.")
        return creds.data["username"]

    async def connect(self) -> str:
        bridge_ip = os.environ.get("HUE_BRIDGE_IP")
        if not bridge_ip:
            return "HUE_BRIDGE_IP is not set. Add it to your .env file and restart."

        logger.info("Attempting Hue bridge connection — press the link button now.")

        try:
            from huesdk import Hue
            username = await asyncio.to_thread(Hue.connect, bridge_ip=bridge_ip)
            await self._save_username(username)
            return "Philips Hue connected successfully!"
        except Exception as e:
            logger.error(f"Hue connect error: {e}")
            return (
                "Couldn't connect to the Hue bridge. "
                "Press the link button on the bridge, then try a home control command again."
            )

    async def _save_username(self, username: str) -> None:
        existing = await ServiceCredentials.find_one(ServiceCredentials.service == "hue")
        if existing:
            existing.data = {"username": username}
            await existing.save()
        else:
            await ServiceCredentials(service="hue", data={"username": username}).insert()
