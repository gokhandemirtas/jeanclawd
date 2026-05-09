import argparse
import asyncio
import os
import sys
import time

from dotenv import load_dotenv
from rich.console import Console

from agents.greeting.greeting_agent import greet
from agents.intent_classifier.intent_classifier import classify
from agents.intent_router.intent_router import route
from db.init import db_init
from utils import logger
from utils.boot_info import boot, check_inference

load_dotenv()

console = Console()


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

	inf_ok, inf_detail = await check_inference()

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

	await asyncio.to_thread(greet)

	while True:
		try:
			user_input = (await asyncio.to_thread(console.input, "❯ ")).strip()
			if not user_input:
				continue
			if user_input.startswith("/"):
				if user_input in ("/quit", "/exit", "/bye"):
					console.print("\nGoodbye! Have a great day!")
					break
				continue

			streamed: list[str] = []

			def on_token(token: str) -> None:
				streamed.append(token)
				console.out(token, end="")

			_start = time.time()
			intent = await asyncio.to_thread(classify, user_input)
			logger.debug(f"Intent: {intent}")

			response = await route(intent, user_input, on_token=on_token)
			_elapsed = time.time() - _start

			if speak:
				from tts.tts import speak as tts_speak

				await asyncio.to_thread(tts_speak, response)

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
