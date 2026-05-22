import asyncio
import base64
import os
import re
import tempfile
import threading

from stt import get_stt_provider
from utils import logger

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # geometric shapes extended
    "\U0001F800-\U0001F8FF"  # supplemental arrows
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess / other
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-a
    "\U00002702-\U000027B0"  # dingbats
    "\U000024C2-\U0001F251"  # enclosed characters
    "]+",
    flags=re.UNICODE,
)


def _clean_for_tts(text: str) -> str:
    return _EMOJI_RE.sub("", text).strip()


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

		transcript = get_stt_provider().transcribe(audio_in)
		logger.debug(f"[api/voice] transcript='{transcript}'")
		job["transcription"] = transcript

		if cancel_event.is_set():
			job.update({"status": "cancelled"})
			return

		from agents.intent_classifier import run_intent_classifier

		response_text = asyncio.run_coroutine_threadsafe(
			run_intent_classifier(transcript, voice_mode=True), loop
		).result()
		job["text"] = response_text

		if cancel_event.is_set():
			job.update({"status": "cancelled"})
			return

		with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
			audio_out = tmp.name
		synthesise(_clean_for_tts(response_text), audio_out, voice_id=voice_id, cancel_event=cancel_event)

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

		transcript = get_stt_provider().transcribe(audio_in)
		logger.debug(f"[api/voice/ws] transcript='{transcript}'")
		send_event({"stage": "transcribed", "text": transcript})

		if cancel_event.is_set():
			return

		from agents.intent_classifier import run_intent_classifier

		response_text = asyncio.run_coroutine_threadsafe(
			run_intent_classifier(transcript, on_token=on_token, voice_mode=True), loop
		).result()

		if cancel_event.is_set():
			return

		with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
			audio_out = tmp.name
		synthesise(_clean_for_tts(response_text), audio_out, voice_id=voice_id, cancel_event=cancel_event)

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
