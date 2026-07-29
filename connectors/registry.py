"""Connector discovery and registry.

Every Python module under :mod:`connectors` is a potential extension.  A
module becomes available when it exposes a concrete ``Connector`` subclass;
there is no central list to update when adding a connector.
"""

import importlib
import inspect
import pkgutil

from connectors.base import Connector
from utils.log import logger

_registry: dict[str, type[Connector]] | None = None


def _discover() -> dict[str, type[Connector]]:
	connectors: dict[str, type[Connector]] = {}
	package = importlib.import_module("connectors")
	for module_info in pkgutil.iter_modules(package.__path__):
		if module_info.name.startswith("_") or module_info.name in {"base", "registry"}:
			continue
		try:
			module = importlib.import_module(f"connectors.{module_info.name}")
		except Exception as exc:
			logger.warning(f"Could not load connector '{module_info.name}': {exc}")
			continue
		for candidate in vars(module).values():
			if (
				inspect.isclass(candidate)
				and candidate.__module__ == module.__name__
				and issubclass(candidate, Connector)
				and candidate is not Connector
				and not inspect.isabstract(candidate)
			):
				service_id = getattr(candidate, "service_id", "")
				if service_id:
					connectors[service_id.lower()] = candidate
	return dict(sorted(connectors.items()))


def connector_types() -> dict[str, type[Connector]]:
	"""Return discovered connector classes keyed by their service ID."""
	global _registry
	if _registry is None:
		_registry = _discover()
	return _registry.copy()


def connector_ids() -> list[str]:
	return list(connector_types())


def get_connector(service_id: str) -> Connector | None:
	connector_type = connector_types().get(service_id.strip().lower())
	return connector_type() if connector_type else None


def reset_registry() -> None:
	"""Clear the cache; useful for tests and development reloads."""
	global _registry
	_registry = None
