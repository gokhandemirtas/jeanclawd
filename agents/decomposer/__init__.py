"""Structured decomposition of compound user requests."""

from agents.decomposer.agent import (
	DecomposedTask,
	DecompositionResult,
	decompose_request,
	decompose_request_sync,
)

__all__ = [
	"DecomposedTask",
	"DecompositionResult",
	"decompose_request",
	"decompose_request_sync",
]
