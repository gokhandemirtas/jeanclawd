"""Abstract base class for skill authentication providers."""

from abc import ABC, abstractmethod


class AuthProvider(ABC):
    """Defines how a skill authenticates with an external service.

    The router calls `is_connected()` before every skill dispatch. If False,
    it calls `connect()` which runs the full interactive auth flow and returns
    a user-facing status message. On success, the router retries the original
    skill automatically.
    """

    service_id: str    # machine-readable key stored in ServiceCredentials (e.g. "spotify")
    display_name: str  # shown to the user (e.g. "Spotify")

    @abstractmethod
    async def is_connected(self) -> bool:
        """Return True if valid credentials exist in the DB and the service is usable."""
        ...

    @abstractmethod
    async def connect(self) -> str:
        """Run the full interactive auth flow.

        Should block until complete (success or failure) and return a
        user-facing message describing the result.
        """
        ...
