import argparse
import asyncio
import os
import select
import sys
import termios
import time
import tty

from dotenv import load_dotenv
from rich.console import Console

from agents.action_resolver.agent import resolve_action_query
from agents.greeting.greeting_agent import greet
from agents.intent_classifier.intent_classifier import classify
from agents.intent_router.intent_router import route
from skills.conversation.skill import resolve_query
from commands import handle_command, _COMMANDS
from db.init import db_init
from utils import logger
from utils.boot_info import boot, check_inference

try:
	from prompt_toolkit import PromptSession
	from prompt_toolkit.completion import WordCompleter

	_completer = WordCompleter(
		sorted(_COMMANDS.keys()) + ["/quit", "/exit", "/bye"],
		sentence=True,
	)
	_prompt_session: PromptSession | None = PromptSession(completer=_completer)
except ImportError:
	_prompt_session = None

load_dotenv()

console = Console()


async def _watch_for_esc(gen_task: "asyncio.Task[str]") -> None:
    """Monitor stdin for an Escape keypress and cancel gen_task if detected."""
    import threading

    stop_event = threading.Event()

    def _blocking_watch() -> None:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                readable, _, _ = select.select([sys.stdin], [], [], 0.05)
                if readable:
                    ch = os.read(fd, 1)
                    if ch == b"\x1b":
                        gen_task.cancel()
                        return
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    try:
        await asyncio.to_thread(_blocking_watch)
    except asyncio.CancelledError:
        stop_event.set()
        raise


async def main(
	debug: bool, emote: bool, speak: bool, track_camera: bool, server: bool
) -> None:
	logger.set_debug(debug)

	if emote:
		from skills.conversation.skill import enable_emote

		enable_emote()
		logger.debug("Emote mode enabled")

	if speak:
		from tts.tts import enable_tts

		enable_tts()
		logger.debug("TTS enabled")

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

	boot(
		connections=connections,
		flags={
			"debug": debug,
			"emote": emote,
			"speak": speak,
			"track_camera": track_camera,
			"server": server,
		},
	)

	if not db_ok:
		console.print("[red]Database unavailable — cannot start.[/red]")
		sys.exit(1)

	from db.preferences import apply_preferences
	from skills.personalizer import start_personalizer
	await apply_preferences()
	await start_personalizer()

	await asyncio.to_thread(greet)

	while True:
		try:
			if _prompt_session is not None:
				user_input = (await _prompt_session.prompt_async("❯ ")).strip()
			else:
				user_input = (await asyncio.to_thread(console.input, "❯ ")).strip()
			if not user_input:
				continue
			if user_input.startswith("/"):
				if user_input in ("/quit", "/exit", "/bye"):
					console.print("\nGoodbye! Have a great day!")
					break
				await handle_command(user_input)
				continue

			streamed: list[str] = []

			def on_token(token: str) -> None:
				streamed.append(token)
				console.out(token, end="")

			async def _generate() -> str:
				nonlocal streamed
				effective_query = await resolve_query(user_input)
				effective_query = await resolve_action_query(effective_query)
				intent = await asyncio.to_thread(classify, effective_query)
				logger.debug(f"Intent: {intent}")
				return await route(intent, effective_query, on_token=on_token)

			_start = time.time()
			gen_task: asyncio.Task[str] = asyncio.create_task(_generate())
			esc_task = asyncio.create_task(_watch_for_esc(gen_task))

			cancelled = False
			response = ""
			try:
				response = await gen_task
			except asyncio.CancelledError:
				cancelled = True
			finally:
				esc_task.cancel()
				try:
					await esc_task
				except asyncio.CancelledError:
					pass

			_elapsed = time.time() - _start

			if cancelled:
				if streamed:
					console.out("")
				console.print("\n[dim]Generation stopped.[/dim]")
				continue

			from skills.journal import append_to_journal
			asyncio.create_task(append_to_journal(user_input, response))

			if speak:
				from tts.tts import speak as tts_speak, active_voice_id
				from agents.persona import get_active_voice_id

				await asyncio.to_thread(tts_speak, response, get_active_voice_id() or active_voice_id)

			if streamed:
				console.out("")
			else:
				logger.newline()
				logger.info(response)
				logger.newline()

			logger.timing(_elapsed)

		except KeyboardInterrupt:
			console.print("\nGoodbye! Have a great day!")
			break
		except EOFError:
			console.print("\nGoodbye! Have a great day!")
			break
		except Exception as e:
			logger.error(f"An error occurred: {e}")
			console.print("\nSorry, I encountered an error. Let's try again.\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--debug", action="store_true", help="Enable debug/warn logging")
	parser.add_argument("--emote", action="store_true", help="Enable emote mode")
	parser.add_argument("--speak", action="store_true", help="Enable TTS output")
	parser.add_argument(
		"--track_camera", action="store_true", help="Enable camera tracking"
	)
	parser.add_argument("--server", action="store_true", help="Start the API server")
	args = parser.parse_args()
	asyncio.run(
		main(
			debug=args.debug,
			emote=args.emote,
			speak=args.speak,
			track_camera=args.track_camera,
			server=args.server,
		)
	)
