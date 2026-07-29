"""Philips Hue bridge pairing connector."""

import asyncio
import os

from connectors.base import Connector
from db.schemas import ServiceCredentials
from utils.log import logger


class HueConnector(Connector):
	service_id = "hue"
	display_name = "Philips Hue"
	description = "Bridge link-button flow"

	async def is_connected(self) -> bool:
		creds = await ServiceCredentials.find_one(ServiceCredentials.service == self.service_id)
		return creds is not None and "username" in creds.data

	async def get_username(self) -> str:
		creds = await ServiceCredentials.find_one(ServiceCredentials.service == self.service_id)
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
		except Exception as exc:
			logger.error(f"Hue connect error: {exc}")
			return "Couldn't connect to the Hue bridge. Press the link button on the bridge, then try a home control command again."

	async def _save_username(self, username: str) -> None:
		existing = await ServiceCredentials.find_one(ServiceCredentials.service == self.service_id)
		if existing:
			existing.data = {"username": username}
			await existing.save()
		else:
			await ServiceCredentials(service=self.service_id, data={"username": username}).insert()
