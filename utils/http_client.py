"""Central HTTP client utility for making async requests across all handlers."""

from typing import Optional, Dict, Any
from utils.log import logger
import aiohttp


class HTTPClient:
    """Reusable HTTP client with persistent session management for async requests.

    This class provides a centralized way for all handlers to make HTTP requests
    with automatic session management and consistent error handling.

    Usage:
        # As a context manager (recommended):
        async with HTTPClient(base_url="https://api.example.com") as client:
            data = await client.get("/endpoint")

        # Manual session management:
        client = HTTPClient()
        await client.start_session()
        data = await client.get("https://api.example.com/endpoint")
        await client.close_session()
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30
    ):
        """Initialize the HTTP client.

        Args:
            base_url: Optional base URL prepended to all relative endpoints
            headers: Optional default headers included in all requests
            timeout: Request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.default_headers = headers or {}
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry - automatically starts session."""
        await self.start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - automatically closes session."""
        await self.close_session()

    async def start_session(self) -> None:
        """Initialize the aiohttp ClientSession if not already started."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.default_headers,
                timeout=self.timeout
            )
            logger.debug("HTTP client session started")

    async def close_session(self) -> None:
        """Close the aiohttp ClientSession if open."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("HTTP client session closed")
            self._session = None

    def _build_url(self, endpoint: str) -> str:
        """Construct full URL from base_url and endpoint.

        Args:
            endpoint: API endpoint (can be relative or absolute URL)

        Returns:
            Complete URL string
        """
        # If endpoint is already a full URL, return as-is
        if endpoint.startswith(("http://", "https://")):
            return endpoint

        # If no base_url set, endpoint must be absolute
        if not self.base_url:
            raise ValueError(
                f"Cannot build URL for relative endpoint '{endpoint}' without base_url"
            )

        # Combine base_url with relative endpoint
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    async def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an async GET request.

        Args:
            endpoint: API endpoint (relative to base_url or absolute URL)
            params: Optional query parameters
            headers: Optional request-specific headers (merged with defaults)
            **kwargs: Additional arguments passed to aiohttp

        Returns:
            Response JSON as dictionary

        Raises:
            aiohttp.ClientError: If the request fails
            ValueError: If endpoint is relative but no base_url is set
        """
        if self._session is None or self._session.closed:
            await self.start_session()

        url = self._build_url(endpoint)
        logger.debug(f"GET {url} | params: {params}")

        async with self._session.get(url, params=params, headers=headers, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an async POST request.

        Args:
            endpoint: API endpoint (relative to base_url or absolute URL)
            data: Optional form data
            json: Optional JSON payload
            headers: Optional request-specific headers (merged with defaults)
            **kwargs: Additional arguments passed to aiohttp

        Returns:
            Response JSON as dictionary

        Raises:
            aiohttp.ClientError: If the request fails
            ValueError: If endpoint is relative but no base_url is set
        """
        if self._session is None or self._session.closed:
            await self.start_session()

        url = self._build_url(endpoint)
        logger.debug(f"POST {url} | json: {bool(json)} | data: {bool(data)}")

        async with self._session.post(url, data=data, json=json, headers=headers, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an async PUT request.

        Args:
            endpoint: API endpoint (relative to base_url or absolute URL)
            data: Optional form data
            json: Optional JSON payload
            headers: Optional request-specific headers (merged with defaults)
            **kwargs: Additional arguments passed to aiohttp

        Returns:
            Response JSON as dictionary

        Raises:
            aiohttp.ClientError: If the request fails
            ValueError: If endpoint is relative but no base_url is set
        """
        if self._session is None or self._session.closed:
            await self.start_session()

        url = self._build_url(endpoint)
        logger.debug(f"PUT {url} | json: {bool(json)} | data: {bool(data)}")

        async with self._session.put(url, data=data, json=json, headers=headers, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def delete(
        self,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an async DELETE request.

        Args:
            endpoint: API endpoint (relative to base_url or absolute URL)
            headers: Optional request-specific headers (merged with defaults)
            **kwargs: Additional arguments passed to aiohttp

        Returns:
            Response JSON as dictionary

        Raises:
            aiohttp.ClientError: If the request fails
            ValueError: If endpoint is relative but no base_url is set
        """
        if self._session is None or self._session.closed:
            await self.start_session()

        url = self._build_url(endpoint)
        logger.debug(f"DELETE {url}")

        async with self._session.delete(url, headers=headers, **kwargs) as response:
            response.raise_for_status()
            return await response.json()
