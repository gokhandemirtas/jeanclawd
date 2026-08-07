"""LM Studio stream instrumentation using the SDK's exact prediction stats."""

from __future__ import annotations

import time
from collections.abc import Iterator

from utils.observability import emit


def iter_with_metrics(model, chat, model_key: str) -> Iterator[str]:
	"""Yield LM Studio response text and emit one exact throughput event."""
	started = time.perf_counter()
	first_content_at: float | None = None
	completed = False
	stream = model.respond_stream(chat)
	try:
		for fragment in stream:
			if fragment.content and first_content_at is None:
				first_content_at = time.perf_counter()
			if fragment.content:
				yield fragment.content
		completed = True
	finally:
		if completed and first_content_at is not None:
			finished = time.perf_counter()
			result = stream.result()
			stats = result.stats
			prompt_tokens = stats.prompt_tokens_count
			completion_tokens = stats.predicted_tokens_count
			generation_seconds = finished - first_content_at
			total_seconds = finished - started
			emit(
				"llm_generation_completed",
				provider="lmstudio",
				model=model_key,
				prompt_tokens=prompt_tokens,
				completion_tokens=completion_tokens,
				total_tokens=stats.total_tokens_count,
				server_ttft_ms=(
					round(stats.time_to_first_token_sec * 1000, 2)
					if stats.time_to_first_token_sec is not None else None
				),
				client_ttft_ms=round((first_content_at - started) * 1000, 2),
				wall_time_ms=round(total_seconds * 1000, 2),
				# Exact generated tokens / time after the first content arrived.
				measured_tokens_per_second=(
					round(completion_tokens / generation_seconds, 2)
					if completion_tokens and generation_seconds > 0 else None
				),
				# LM Studio's own decode throughput is best for model comparisons.
				server_tokens_per_second=stats.tokens_per_second,
				end_to_end_tokens_per_second=(
					round(completion_tokens / total_seconds, 2)
					if completion_tokens and total_seconds > 0 else None
				),
				stop_reason=stats.stop_reason,
			)
