import os
import tempfile

from utils.messages import HumanMessage, SystemMessage
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters

from utils.log import logger

_TELEGRAM_MAX_CHARS = 4096
_tts_enabled: bool = False


def enable_tts(enabled: bool = True) -> None:
    global _tts_enabled
    _tts_enabled = enabled


def _allowed_chat_ids() -> set[int]:
    raw = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "")
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


async def _fit_response(text: str) -> str:
    """Summarise the response if it exceeds Telegram's character limit."""
    if len(text) <= _TELEGRAM_MAX_CHARS:
        return text

    from agents.agent_factory import create_llm
    logger.debug(f"Telegram response too long ({len(text)} chars), summarising")
    llm = create_llm(temperature=0.3, max_tokens=512)
    result = await llm.ainvoke([
        SystemMessage(content="Summarise the following text concisely. Preserve the key facts and conclusions. Output must be under 4000 characters."),
        HumanMessage(content=text),
    ])
    summary = result.content.strip()
    # Hard fallback: truncate if summarisation itself is still too long
    return summary[:_TELEGRAM_MAX_CHARS] if len(summary) > _TELEGRAM_MAX_CHARS else summary


async def _transcribe_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str | None:
    """Download and transcribe a voice or audio message. Returns transcribed text or None on failure."""
    import asyncio

    from voicebox.client import transcribe

    file = await (update.message.voice or update.message.audio).get_file()

    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
        tmp_path = f.name

    try:
        await file.download_to_drive(tmp_path)
        text = await asyncio.to_thread(transcribe, tmp_path)
        logger.debug(f"STT transcription: {text}")
        return text
    except Exception as e:
        logger.error(f"STT transcription failed: {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


async def _process_query(update: Update, context: ContextTypes.DEFAULT_TYPE, user_text: str, audio_reply: bool = False) -> None:
	"""Run intent classification or planner on user_text and reply."""
	from agents.intent_classifier import run_intent_classifier
	from utils.observability import request_context

	chat_id = update.effective_chat.id
	await context.bot.send_chat_action(chat_id=chat_id, action="typing")

	with request_context("telegram", "message", chat_id=chat_id, audio_reply=audio_reply):
		try:
			if user_text.startswith("@planner"):
				from agents.planner import run_planner
				from utils.inference_gate import foreground_inference
				from utils.langmem_service import record_interaction

				task = user_text[len("@planner"):].strip()
				if not task:
					await update.message.reply_text("Please provide a task. Example: @planner check the weather and set a reminder for 8am")
					return
				async with foreground_inference():
					response = await run_planner(task)
				await record_interaction(task, response, intent="PLANNER")
			else:
				response = await run_intent_classifier(user_text)

			if audio_reply or _tts_enabled:
				await _reply_audio(update, response)
			else:
				await update.message.reply_text(await _fit_response(response))
		except Exception as e:
			logger.error(f"Telegram handler error: {e}")
			await update.message.reply_text("Sorry, something went wrong. Please try again.")


async def _handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
	"""Run /vision for Telegram photos and image documents."""
	chat_id = update.effective_chat.id
	allowed = _allowed_chat_ids()
	if allowed and chat_id not in allowed:
		logger.warning(f"Telegram: rejected message from unauthorized chat_id={chat_id}")
		return

	try:
		from skills.vision import handle_vision

		if update.message.photo:
			telegram_file = await update.message.photo[-1].get_file()
			filename = "telegram-photo.jpg"
		else:
			telegram_file = await update.message.document.get_file()
			filename = update.message.document.file_name or "telegram-image"

		with tempfile.NamedTemporaryFile(suffix=".image", delete=False) as f:
			tmp_path = f.name
		await telegram_file.download_to_drive(tmp_path)
		try:
			with open(tmp_path, "rb") as image_file:
				image_bytes = image_file.read()
		finally:
			os.remove(tmp_path)

		caption = (update.message.caption or "").strip()
		prompt = caption.removeprefix("/vision").strip() or None
		await context.bot.send_chat_action(chat_id=chat_id, action="typing")
		description = await handle_vision(image_bytes, prompt=prompt, filename=filename)
		await update.message.reply_text(await _fit_response(description))
	except Exception as e:
		logger.error(f"Telegram image handler error: {e}")
		await update.message.reply_text("Sorry, I couldn't analyse that image.")


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    allowed = _allowed_chat_ids()

    if allowed and chat_id not in allowed:
        logger.warning(f"Telegram: rejected message from unauthorized chat_id={chat_id}")
        return

    user_text = update.message.text
    logger.debug(f"Telegram [{chat_id}]: {user_text}")
    await _process_query(update, context, user_text)


async def _handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    allowed = _allowed_chat_ids()

    if allowed and chat_id not in allowed:
        logger.warning(f"Telegram: rejected message from unauthorized chat_id={chat_id}")
        return

    logger.debug(f"Telegram [{chat_id}]: received audio message")
    user_text = await _transcribe_message(update, context)
    if not user_text:
        await update.message.reply_text("Sorry, I couldn't understand the audio.")
        return

    await _process_query(update, context, user_text, audio_reply=True)


async def _reply_audio(update: Update, text: str) -> None:
    import asyncio

    from voicebox.client import synthesize

    try:
        wav_bytes = await asyncio.to_thread(synthesize, text, None, False)
        await update.message.reply_audio(audio=wav_bytes, title="Response")
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        await update.message.reply_text(await _fit_response(text))


async def start_telegram_bot() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN not set — Telegram bot will not start")
        return

    app = ApplicationBuilder().token(token).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, _handle_audio))
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, _handle_image))

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

    return "Telegram bot started and polling"
