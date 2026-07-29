"""Backward-compatible name for the connector contract.

Skills historically imported ``AuthProvider`` from this module.  Keep that
API stable while making connectors the shared extension point for both skills
and the ``/connect`` command.
"""

from connectors.base import Connector


AuthProvider = Connector

__all__ = ["AuthProvider"]
