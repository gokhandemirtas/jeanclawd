import asyncio
import base64
import os
import tempfile
import threading

import whisper

from utils import logger


def process_voice_job(
	job: dict,
	audio_in: str,
	loop: asyncio.AbstractEventLoop,
	voice_id: str,
) -> None:
	"""Blocking pipeline: Whisper → intent classifier → TTS. Runs in thread pool."""
	from tts.tts import synthesise

	cancel_event: threading.Event = job["cancel_event"]
	audio_out = None
	try:
		if cancel_event.is_set():
			job.update({"status": "cancelled"})
			return

		transcript = whisper.load_model("base").transcribe(audio_in)["text"].strip()
		logger.debug(f"[api/voice] transcript='{transcript}'")
		job["transcription"] = transcript

		if cancel_event.is_set():
			job.update({"status": "cancelled"})
			return

		from agents.intent_classifier import run_intent_classifier

		response_text = asyncio.run_coroutine_threadsafe(
			run_intent_classifier(transcript), loop
		).result()
		job["text"] = response_text

		if cancel_event.is_set():
			job.update({"status": "cancelled"})
			return

		with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
			audio_out = tmp.name
		synthesise(response_text, audio_out, voice_id=voice_id, cancel_event=cancel_event)

		if cancel_event.is_set():
			job.update({"status": "cancelled"})
			return

		with open(audio_out, "rb") as f:
			audio_b64 = base64.b64encode(f.read()).decode()

		job.update({"status": "done", "transcription": transcript, "text": response_text, "audio": audio_b64})

	except Exception as e:
		logger.error(f"[api/voice] pipeline failed: {e}")
		job.update({"status": "error", "error": "Something went wrong. Please try again."})
	finally:
		for path in (audio_in, audio_out):
			if path and os.path.exists(path):
				os.remove(path)


def process_voice_ws(
	audio_in: str,
	loop: asyncio.AbstractEventLoop,
	voice_id: str,
	cancel_event: threading.Event,
	send_event,
	on_token,
) -> None:
	"""Blocking pipeline: Whisper → LLM → TTS for the WebSocket path."""
	from tts.tts import synthesise

	audio_out = None
	try:
		if cancel_event.is_set():
			return

		transcript = whisper.load_model("base").transcribe(audio_in)["text"].strip()
		logger.debug(f"[api/voice/ws] transcript='{transcript}'")
		send_event({"stage": "transcribed", "text": transcript})

		if cancel_event.is_set():
			return

		from agents.intent_classifier import run_intent_classifier

		response_text = asyncio.run_coroutine_threadsafe(
			run_intent_classifier(transcript, on_token=on_token), loop
		).result()

		if cancel_event.is_set():
			return

		with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
			audio_out = tmp.name
		synthesise(response_text, audio_out, voice_id=voice_id, cancel_event=cancel_event)

		if cancel_event.is_set():
			return

		with open(audio_out, "rb") as f:
			audio_b64 = base64.b64encode(f.read()).decode()

		send_event({"stage": "done", "text": response_text, "audio": audio_b64})

	except Exception as e:
		logger.error(f"[api/voice/ws] pipeline failed: {e}")
		send_event({"stage": "error", "error": "Something went wrong. Please try again."})
	finally:
		for path in (audio_in, audio_out):
			if path and os.path.exists(path):
				os.remove(path)
