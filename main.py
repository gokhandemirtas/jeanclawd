import argparse
import asyncio
import logging
import os
import signal
import sys
import time

from dotenv import load_dotenv

logging.getLogger("lmstudio").setLevel(logging.CRITICAL)
from rich.console import Console

from agents.action_resolver.agent import resolve_action_query
from agents.generalist.router import generalist_route
from agents.greeting.greeting_agent import greet
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


async def main(
	debug: bool, emote: bool, speak: bool, track_camera: bool, server: bool
) -> None:
	logger.set_debug(debug)
	logger.init_file_logging("logs/app.log")

	loop = asyncio.get_running_loop()
	active_gen_task: asyncio.Task | None = None
	last_sigint: float = 0.0

	def _on_sigint() -> None:
		nonlocal last_sigint, active_gen_task
		now = time.time()
		if active_gen_task is not None and not active_gen_task.done():
			active_gen_task.cancel()
		else:
			if now - last_sigint < 1.5:
				console.print("\nGoodbye! Have a great day!")
				sys.exit(0)
			else:
				console.print("\n[dim]Press Ctrl+C again to exit.[/dim]")
		last_sigint = now

	loop.add_signal_handler(signal.SIGINT, _on_sigint)

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
				from skills.conversation.skill import get_recent_history
				effective_query = await resolve_query(user_input)
				effective_query = await resolve_action_query(effective_query)
				intent = await generalist_route(effective_query, get_recent_history(n=3))
				return await route(intent, effective_query, on_token=on_token)

			_start = time.time()
			active_gen_task = asyncio.create_task(_generate())

			cancelled = False
			response = ""
			try:
				response = await active_gen_task
			except asyncio.CancelledError:
				cancelled = True
			finally:
				active_gen_task = None

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
