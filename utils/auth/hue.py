"""Compatibility import for the relocated Philips Hue connector."""

from connectors.hue import HueConnector

HueAuthProvider = HueConnector

__all__ = ["HueAuthProvider"]
