import argparse
import asyncio
import logging
import os
import sys
import threading
import time

from dotenv import load_dotenv

logging.getLogger("lmstudio").setLevel(logging.CRITICAL)
logging.getLogger("AsyncWebsocketHandler").setLevel(logging.CRITICAL)
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from agents.action_resolver.agent import resolve_action_query
from agents.generalist.router import generalist_route
from agents.intent_router.intent_router import route
from skills.conversation.skill import generate_greeting, resolve_query
from commands import handle_command, _COMMANDS
from db.init import db_init
from utils import logger
from utils.boot_info import boot, check_inference
from utils.task_queue import drain, enqueue, pending_count, worker
from utils.pending_plan import (
	clear_pending,
	is_cancellation,
	is_confirmation,
	set_pending,
	take_pending,
)

try:
	from prompt_toolkit import PromptSession
	from prompt_toolkit.completion import WordCompleter

	_completer = WordCompleter(
		sorted(_COMMANDS.keys()) + ["/quit"],
		sentence=True,
	)
	_prompt_session: PromptSession | None = PromptSession(completer=_completer)
except ImportError:
	_prompt_session = None

load_dotenv()

console = Console()
_spinner_console = Console(stderr=True, force_terminal=True)

# Whimsical one-word "thinking" verbs that rotate while the spinner is
# pending. Real words and made-up ones, mixed for flavour. Order is the
# display order — it cycles from the top.
_THINKING_WORDS: list[str] = [
	"flummoxing",
	"jimmijamming",
	"bumbershooting",
	"kerfuffling",
	"galumphing",
	"razzmatazzing",
	"whirligigging",
	"widdershining",
	"flibbertigibbeting",
	"thingamajigging",
	"fribbling",
	"fudgulating",
	"boggling",
	"coggling",
	"mumbojumboing",
]
_WORD_INTERVAL_S = 2.0


class _ProgressSpinner:
	"""Threaded progress spinner that renders to stderr.

	The spinner runs in a background thread so the main async loop can keep
	processing (routing, queued tasks, etc.). `stop()` blocks until the
	`rich.live.Live` context has fully exited and the spinner line is
	cleared — safe to call from a callback right before writing the first
	token to stdout, with no garbled output.

	Without an explicit `message`, the spinner text rotates through a list
	of whimsical "thinking" verbs, changing every `_WORD_INTERVAL_S`
	seconds. Pass a `message` to use a static label instead (for call
	sites that need a specific status).
	"""

	def __init__(self, message: str | None = None) -> None:
		self._message = message
		self._stop_event = threading.Event()
		self._thread: threading.Thread | None = None

	def start(self) -> None:
		if self._thread is not None and self._thread.is_alive():
			return
		self._stop_event.clear()
		self._thread = threading.Thread(target=self._spin, daemon=True)
		self._thread.start()

	def _spin(self) -> None:
		def _renderable(idx: int) -> Spinner:
			if self._message is not None:
				word = self._message
			else:
				word = _THINKING_WORDS[idx % len(_THINKING_WORDS)]
			return Spinner("dots", text=Text(word, style="dim"))

		idx = 0
		with Live(
			_renderable(idx),
			console=_spinner_console,
			transient=True,
			refresh_per_second=10,
		) as live:
			while not self._stop_event.wait(_WORD_INTERVAL_S):
				idx += 1
				live.update(_renderable(idx))

	def stop(self) -> None:
		self._stop_event.set()
		if self._thread is not None:
			self._thread.join(timeout=1.0)

	def __enter__(self) -> "_ProgressSpinner":
		self.start()
		return self

	def __exit__(self, *exc_info) -> None:
		self.stop()


async def main(
	debug: bool,
	emote: bool,
	speak: bool,
	track_camera: bool,
	server: bool,
	listen: bool,
	telegram: bool = False,
	session_id: str | None = None,
) -> None:
	logger.set_debug(debug)
	logger.init_file_logging("logs/app.log")

	asyncio.create_task(worker())

	if emote:
		from skills.conversation.skill import enable_emote

		enable_emote()
		logger.debug("Emote mode enabled")

	if speak:
		from tts.tts import enable_tts, set_spinner

		enable_tts()
		set_spinner(False)  # spinner corrupts streamed text in CLI mode
		logger.debug("TTS enabled")

	if listen:
		from stt import get_stt_provider
		from stt.wakeword import preload as preload_wakeword
		await asyncio.to_thread(get_stt_provider)
		await asyncio.to_thread(preload_wakeword)
		logger.debug("STT voice input enabled (wake word: hey jarvis)")

	from agents.persona import set_io_capabilities
	set_io_capabilities(speak=speak, listen=listen)

	if track_camera:
		from skills.camera_tracking.skill import start_tracking

		await asyncio.to_thread(start_tracking)
		logger.debug("Camera tracking started")

	api_url: str | None = None
	if server:
		from api.server import start_api_server

		_api_host, _api_port = "0.0.0.0", 8000
		await start_api_server(host=_api_host, port=_api_port)
		api_url = f"http://{_api_host}:{_api_port}"
		logger.debug("API server started")

	from providers import backend_label as _backend_label
	inf_ok, inf_detail = await check_inference()
	inf_detail = f"{_backend_label()} — {inf_detail}"

	db_ok, db_detail = False, ""
	try:
		mongo_host = os.getenv("MONGODB_HOST", "localhost")
		mongo_port = os.getenv("MONGODB_PORT", "27017")
		db_detail = f"mongodb://{mongo_host}:{mongo_port}"
		await db_init()
		db_ok = True
	except Exception as exc:
		db_detail = str(exc)

	connections: dict[str, tuple[bool, str]] = {
		"Inference": (inf_ok, inf_detail),
		"Database": (db_ok, db_detail),
	}
	if api_url:
		connections["API"] = (True, api_url)

	if not db_ok:
		console.print("[red]Database unavailable — cannot start.[/red]")
		sys.exit(1)

	if telegram:
		from skills.telegram import start_telegram_bot

		telegram_status = await start_telegram_bot()
		connections["Telegram"] = (
			bool(telegram_status), telegram_status or "TELEGRAM_BOT_TOKEN not set"
		)

	from db.preferences import apply_preferences
	await apply_preferences()
	from utils.session_service import initialize_session, resume_session
	try:
		if session_id:
			await resume_session(session_id)
		else:
			await initialize_session()
	except ValueError:
		console.print(f"[red]Could not resume session: {session_id}[/red]")
		return
	from utils.memory_service import start_memory_worker
	start_memory_worker()

	from agents.persona import get_active_persona_id, get_active_voice_id
	boot(
		connections=connections,
		flags={
			"debug": debug,
			"emote": emote,
			"speak": speak,
			"listen": listen,
			"track_camera": track_camera,
			"server": server,
			"telegram": telegram,
		},
		persona={
			"Persona": get_active_persona_id(),
			"Voice": get_active_voice_id() or "—",
		},
	)

	greeting = await generate_greeting()
	console.print(greeting)

	while True:
		try:
			if listen:
				from stt import get_stt_provider
				from stt.wakeword import listen_for_wakeword
				from stt.recorder import record_until_silence
				console.print("[dim]Listening… (say 'hey jarvis')[/dim]")
				await asyncio.to_thread(listen_for_wakeword)
				console.print("[dim]🎙  Recording…[/dim]", end="\r")
				audio_path, too_short = await asyncio.to_thread(record_until_silence)
				if too_short:
					os.unlink(audio_path)
					continue
				console.print("[dim]Transcribing…[/dim]", end="\r")
				try:
					user_input = await asyncio.to_thread(get_stt_provider().transcribe, audio_path)
				finally:
					os.unlink(audio_path)
				if not user_input:
					continue
				# Guard against Whisper hallucinations (single word repeated many times)
				words = user_input.split()
				if len(words) > 6 and len(set(w.lower() for w in words)) <= 2:
					continue
				console.print(f"[cyan]You:[/cyan] {user_input}")
			elif _prompt_session is not None:
				user_input = (await _prompt_session.prompt_async("❯ ")).strip()
			else:
				user_input = (await asyncio.to_thread(console.input, "❯ ")).strip()
			if not user_input:
				continue
			if user_input.startswith("/"):
				if user_input in ("/quit",):
					_print_resume_hint()
					break
				await handle_command(user_input)
				continue

			with _ProgressSpinner() as spinner:
				streamed: list[str] = []
				streamed_visible = False

				def on_token(token: str) -> None:
					nonlocal streamed_visible
					if not streamed:
						spinner.stop()
						print("\033[36m", end="", flush=True)  # cyan on first token
					streamed.append(token)
					streamed_visible = True
					print(token, end="", flush=True)

				loop = asyncio.get_running_loop()
				done: asyncio.Future[str] = loop.create_future()
				resolved_intent: str | None = None

				async def _queued_generate() -> None:
					nonlocal resolved_intent
					from skills.conversation.skill import get_recent_history, consume_pending_intent
					from utils.inference_gate import foreground_inference
					try:
						async with foreground_inference():
							effective_query = await resolve_query(user_input)
							effective_query = await resolve_action_query(effective_query)
							planned = take_pending()
							if planned is not None:
								if is_cancellation(user_input):
									result = "Cancelled the pending requests."
								elif is_confirmation(user_input):
									parts: list[str] = []
									for index, task in enumerate(planned, 1):
										try:
											resolved_intent = task.intent
											answer = await route(task.intent, task.query)
											parts.append(f"{index}. {answer}")
										except Exception as exc:
											logger.exception(f"[decomposer] Task {index} failed")
											parts.append(f"{index}. Failed: {exc}")
									result = "\n\n".join(parts)
								else:
									clear_pending()
									result = "I cancelled the pending requests because the confirmation response was unclear."
							else:
								from agents.decomposer import decompose_request, extract_batch_request
								batch_query = extract_batch_request(effective_query)
								decomposition = await decompose_request(batch_query) if batch_query else None
								if decomposition is not None and decomposition.is_compound:
									set_pending(decomposition.tasks)
									lines = [f"I found {len(decomposition.tasks)} requests:"]
									lines.extend(
										f"{i}. {task.description}"
										for i, task in enumerate(decomposition.tasks, 1)
									)
									lines.append("\nProcess these in order? [y/n]")
									result = "\n".join(lines)
								else:
									# A single task deliberately goes through the existing router;
									# decomposition does not replace intent classification.
									from agents.intent_classifier.intent_classifier import _keyword_classify
									shortcut = _keyword_classify(effective_query)
									pending = consume_pending_intent()
									if shortcut:
										if pending:
											logger.debug(f"[router] explicit {shortcut} overrides pending intent lock: {pending}")
										intent = shortcut
										confidence = 1.0
									elif pending:
										logger.debug(f"[router] pending intent lock: {pending}")
										intent = pending
										confidence = 1.0
									else:
										intent, confidence = await generalist_route(effective_query, get_recent_history(n=3))
										if confidence < 0.75 and intent != "CHAT":
											logger.debug(f"[router] low confidence ({confidence:.2f}) for {intent}, falling back to CHAT")
											intent = "CHAT"
									resolved_intent = intent
									result = await route(intent, effective_query, on_token=on_token)
						if not done.done():
							done.set_result(result)
					except Exception as e:
						if not done.done():
							done.set_exception(e)

				enqueue("query", _queued_generate())

				_start = time.time()
				response = await done
				_elapsed = time.time() - _start

				if streamed_visible:
					print("\033[0m")  # reset color + newline
				if not streamed_visible or "".join(streamed).strip() != response.strip():
					spinner.stop()  # ensure cleared before the logged fallback
					logger.newline()
					logger.info(response)
					logger.newline()

			# Interaction persistence is cheap and must not depend on the old queue.
			await _record_interaction(user_input, response, intent=resolved_intent)

			if speak:
				from tts.expressions import decorate_for_speech
				from tts.tts import speak as tts_speak, active_voice_id
				from agents.persona import get_active_voice_id

				voice_spinner = _ProgressSpinner("Synthesizing voice…")
				voice_spinner.start()
				try:
					# News is already formatted by its dedicated skill. Read the exact
					# displayed briefing rather than sending it through persona decoration.
					if resolved_intent == "NEWSFEED":
						spoken_response = response
						logger.debug("[tts] NEWSFEED: speaking displayed response verbatim")
					else:
						spoken_response = await asyncio.to_thread(decorate_for_speech, response)
					await asyncio.to_thread(
						tts_speak,
						spoken_response,
						get_active_voice_id() or active_voice_id,
					)
				finally:
					voice_spinner.stop()

			logger.timing(_elapsed)

		except KeyboardInterrupt:
			count = pending_count()
			if count == 0:
				_print_resume_hint()
				break
			console.print(f"\n[yellow]{count} task(s) queued. Clear queue? [y/n][/yellow] ", end="")
			try:
				answer = input().strip().lower()
			except (EOFError, KeyboardInterrupt):
				_print_resume_hint()
				break
			if answer == "y":
				removed = drain()
				console.print(f"[dim]Cleared {removed} task(s). Press Ctrl+C to exit.[/dim]")
		except EOFError:
			_print_resume_hint()
			break
		except Exception as e:
			logger.error(f"An error occurred: {e}")
			console.print("\nSorry, I encountered an error. Let's try again.\n")


def _print_resume_hint() -> None:
	from utils.session_service import get_active_session_id

	session_id = get_active_session_id()
	if session_id:
		console.print(
			f"\nGoodbye! Resume this session with:\n"
			f"[cyan]uv run main.py --session {session_id}[/cyan]"
		)
	else:
		console.print("\nGoodbye! Have a great day!")


async def _record_interaction(question: str, answer: str, *, intent: str | None = None) -> None:
	from utils.memory_service import record_interaction
	await record_interaction(question, answer, intent=intent)


def _keepalive_loop(forwarded_args: list[str]) -> None:
	"""Restart loop that inherits the terminal — required for the interactive CLI.

	Supervisord captures stdin and breaks prompt_toolkit; subprocess.run inherits
	the parent's stdio so the REPL works normally. Exits cleanly on returncode 0
	(Ctrl+C / /quit); aborts after 3 rapid crashes (< 5 s uptime) to avoid
	looping forever on unrecoverable errors like a bound port.
	"""
	import subprocess
	import pathlib

	# If supervisord is running and managing this process, shut it down first
	# to avoid two keepalive mechanisms fighting over the same port.
	sock = pathlib.Path(__file__).parent / "logs" / "supervisor.sock"
	if sock.exists():
		supervisorctl = pathlib.Path(__file__).parent / ".venv" / "bin" / "supervisorctl"
		conf = pathlib.Path(__file__).parent / "supervisord.conf"
		subprocess.run([str(supervisorctl), "-c", str(conf), "shutdown"],
			capture_output=True)
		time.sleep(1)

	_RAPID_CRASH_THRESHOLD = 5.0
	_MAX_RAPID_CRASHES = 3

	cmd = [sys.executable, __file__] + forwarded_args
	rapid_crashes = 0

	while True:
		started_at = time.time()
		result = subprocess.run(cmd)
		uptime = time.time() - started_at

		if result.returncode == 0:
			break

		if uptime < _RAPID_CRASH_THRESHOLD:
			rapid_crashes += 1
			if rapid_crashes >= _MAX_RAPID_CRASHES:
				print(f"\n[keepalive] Crashed {rapid_crashes} times in under {_RAPID_CRASH_THRESHOLD}s — giving up.")
				sys.exit(1)
		else:
			rapid_crashes = 0

		print(f"\n[keepalive] Exited with code {result.returncode} — restarting in 2s…")
		time.sleep(2)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--debug", action="store_true", help="Enable debug/warn logging")
	parser.add_argument("--emote", action="store_true", help="Enable emote mode")
	parser.add_argument("--speak", action="store_true", help="Enable TTS output")
	parser.add_argument(
		"--track_camera", action="store_true", help="Enable camera tracking"
	)
	parser.add_argument("--server", action="store_true", help="Start the API server")
	parser.add_argument("--listen", action="store_true", help="Enable microphone voice input (push-to-talk)")
	parser.add_argument("--telegram", action="store_true", help="Enable the Telegram bot")
	parser.add_argument("--keepalive", action="store_true", help="Run under supervisord (auto-restart on crash)")
	parser.add_argument("--session", help="Resume an existing session by UUID")
	args = parser.parse_args()

	if args.keepalive:
		forwarded = [a for a in sys.argv[1:] if a != "--keepalive"]
		_keepalive_loop(forwarded)  # blocks here, restarts on crash, exits on clean quit
		sys.exit(0)

	asyncio.run(
		main(
			debug=args.debug,
			emote=args.emote,
			speak=args.speak,
			track_camera=args.track_camera,
			server=args.server,
			listen=args.listen,
			telegram=args.telegram,
			session_id=args.session,
		)
	)
