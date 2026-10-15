import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

logging.getLogger("lmstudio").setLevel(logging.CRITICAL)
logging.getLogger("AsyncWebsocketHandler").setLevel(logging.CRITICAL)

from db.init import db_init
from utils import logger
from utils.task_queue import worker

load_dotenv()


async def main(
    emote: bool,
    speak: bool,
    track_camera: bool,
    server: bool,
    telegram: bool = False,
    session_id: str | None = None,
) -> None:
    from pathlib import Path

    for runtime_dir in ("logs", "memory", "storage", "introspections"):
        Path(runtime_dir).mkdir(parents=True, exist_ok=True)
    logger.init_file_logging("logs/app.log")

    asyncio.create_task(worker())

    if emote:
        from skills.conversation.skill import enable_emote
        enable_emote()

    if speak:
        from voicebox.client import enable_tts
        enable_tts()

    from agents.persona import set_io_capabilities
    set_io_capabilities(speak=speak, listen=False)

    if track_camera:
        from skills.camera_tracking.skill import start_tracking
        await asyncio.to_thread(start_tracking)

    if server:
        from api.server import start_api_server
        await start_api_server(host="0.0.0.0", port=8000)

    try:
        await db_init()
    except Exception as exc:
        logger.error(f"Database unavailable — cannot start: {exc}")
        sys.exit(1)

    if telegram:
        from skills.telegram_skill import start_telegram_bot
        await start_telegram_bot()

    from db.preferences import apply_preferences
    await apply_preferences()

    from utils.session_service import initialize_session, resume_session
    try:
        if session_id:
            await resume_session(session_id)
        else:
            await initialize_session()
    except ValueError:
        logger.error(f"Could not resume session: {session_id}")
        return

    await asyncio.Event().wait()

    from utils.langmem_service import close_memory_store
    await close_memory_store()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emote", action="store_true")
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--track_camera", action="store_true")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--session", help="Resume an existing session by UUID")
    args = parser.parse_args()

    asyncio.run(
        main(
            emote=args.emote,
            speak=args.speak,
            track_camera=args.track_camera,
            server=args.server,
            telegram=args.telegram,
            session_id=args.session,
        )
    )
