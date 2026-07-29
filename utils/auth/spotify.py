"""Compatibility import for the relocated Spotify connector."""

from connectors.spotify import SpotifyConnector

SpotifyAuthProvider = SpotifyConnector

__all__ = ["SpotifyAuthProvider"]
