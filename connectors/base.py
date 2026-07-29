"""Contract implemented by services that can be connected through the CLI."""

from abc import ABC, abstractmethod


class Connector(ABC):
	"""Base class for an interactive external-service connection.

	Drop-in connectors only need to provide these metadata fields and implement
	``is_connected`` and ``connect``.  The instance can expose additional
	service-specific methods for skills to use after authentication.
	"""

	service_id: str
	display_name: str
	description: str = ""

	@abstractmethod
	async def is_connected(self) -> bool:
		"""Return whether usable credentials are already available."""
		...

	@abstractmethod
	async def connect(self) -> str:
		"""Run the interactive connection flow and return a user-facing result."""
		...
