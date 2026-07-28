"""Vision Language Model integration.

Sends an image to MODEL_VLM via the OpenAI-compatible chat completions endpoint
using the standard vision message format (base64-encoded image in content array).
Always uses the OpenAI-compatible HTTP path regardless of INFERENCE_PROVIDER,
since that format is universal across LM Studio, llama.cpp, and hosted VLMs.
"""

import asyncio
import base64
import os

import requests

from utils.log import logger

_DEFAULT_PROMPT = (
	"Describe everything visible in this image. Be specific about objects, people, "
	"text, colours, and spatial relationships. Do not speculate. Keep it to 1-2 paragraphs."
)


_MAX_DIMENSION = 800


def _detect_mime(data: bytes) -> str:
	if data[:2] == b"\xff\xd8":
		return "image/jpeg"
	if data[:8] == b"\x89PNG\r\n\x1a\n":
		return "image/png"
	if data[:6] in (b"GIF87a", b"GIF89a"):
		return "image/gif"
	if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
		return "image/webp"
	return "image/jpeg"


def _resize_if_needed(data: bytes) -> bytes:
	"""Scale down to _MAX_DIMENSION on the longest side, preserving aspect ratio."""
	from PIL import Image
	import io

	img = Image.open(io.BytesIO(data))
	w, h = img.size
	if w <= _MAX_DIMENSION and h <= _MAX_DIMENSION:
		return data

	scale = _MAX_DIMENSION / max(w, h)
	new_size = (int(w * scale), int(h * scale))
	img = img.resize(new_size, Image.LANCZOS)
	logger.debug(f"[vlm] Resized {w}×{h} → {new_size[0]}×{new_size[1]}")

	buf = io.BytesIO()
	fmt = img.format or "JPEG"
	img.save(buf, format=fmt)
	return buf.getvalue()


def describe_image_sync(image_bytes: bytes, prompt: str = _DEFAULT_PROMPT) -> str:
	model = os.environ.get("MODEL_VLM")
	if not model:
		raise ValueError("MODEL_VLM environment variable is not set")

	base = os.environ.get("MODEL_HOST", "http://localhost:1234/v1").rstrip("/")
	headers = {
		"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'lm-studio')}",
		"Content-Type": "application/json",
	}

	image_bytes = _resize_if_needed(image_bytes)
	mime = _detect_mime(image_bytes)
	b64 = base64.b64encode(image_bytes).decode()

	logger.debug(f"[vlm] Sending {len(image_bytes):,} bytes ({mime}) to {model}")
	payload = {
		"model": model,
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": prompt},
					{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
				],
			}
		],
		"max_tokens": 512,
		"temperature": 0.2,
	}

	logger.debug(f"[vlm] Sending {len(image_bytes):,} bytes ({mime}) to {model}")
	resp = requests.post(
		f"{base}/chat/completions", headers=headers, json=payload, timeout=60
	)
	resp.raise_for_status()
	description = resp.json()["choices"][0]["message"]["content"].strip()
	logger.debug(f"[vlm] Description: {description[:120]}…")
	return description


async def describe_image(image_bytes: bytes, prompt: str = _DEFAULT_PROMPT) -> str:
	"""Async wrapper — runs the blocking HTTP call in a thread."""
	return await asyncio.to_thread(describe_image_sync, image_bytes, prompt)
