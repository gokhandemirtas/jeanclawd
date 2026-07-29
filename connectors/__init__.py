"""Extensible external-service connectors.

Add a module containing a concrete :class:`connectors.base.Connector` subclass
to make it appear in ``/connect`` and its autocomplete suggestions.
"""

from connectors.base import Connector
from connectors.registry import connector_ids, connector_types, get_connector

__all__ = ["Connector", "connector_ids", "connector_types", "get_connector"]
