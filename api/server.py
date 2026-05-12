"""jeanclawd HTTP API — serves the mobile app over Tailscale.

Endpoints:
  POST /query              — text query through the intent classifier (synchronous)
  WS   /query/ws           — text query with streaming status + token events
  GET  /tts/voices         — list available TTS voice IDs and the default
  GET  /personas           — list available persona IDs
  GET  /persona            — get the active persona
  POST /persona            — set the active persona (body: {"persona": "<id>"})
  WS   /voice/ws           — WebSocket voice round-trip (audio in → transcribed + done events out)
  POST /voice              — audio file → synthesised response (optional ?voice_id=)
  GET  /voice/{job_id}     — poll for voice job result (pending | done | error)
  DEL  /voice/{job_id}     — cancel an in-flight voice job
  WS   /compact            — compact a day's conversations, then auto-writes journal; pushes pending → complete/error (?date=YYYY-MM-DD)
  GET  /conversations      — list all dates that have conversation records (newest first)
  GET  /conversation/{date} — exchanges + compact state for a specific date (YYYY-MM-DD)
  GET  /journal            — list journal entries (?date= or ?limit=)
  GET  /plans              — list saved planner plans
  GET  /notes              — list all notes from MongoDB
  GET  /reminders          — list reminders from MongoDB
  GET  /openapi.json       — OpenAPI 3.0 spec (generated from skills registry)
  GET  /docs               — Swagger UI
"""

import asyncio
import base64
import json
import os
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from aiohttp import WSMsgType, web

from tts.pipeline import process_voice_job, process_voice_ws
from utils import logger

# ---------------------------------------------------------------------------
# Voice job queue
# ---------------------------------------------------------------------------

_voice_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice")
_voice_jobs: dict[str, dict] = {}  # job_id → {status, result}


async def handle_query(request: web.Request) -> web.Response:
	data = await request.json()
	query = data.get("query", "").strip()
	if not query:
		return web.json_response({"error": "query is required"}, status=400)

	if query.startswith("/planner"):
		task = query[len("/planner") :].strip()
		if not task:
			return web.json_response(
				{"error": "task is required. Usage: /planner <task>"}, status=400
			)
		from skills.planner import run_planner

		response = await run_planner(task)
		return web.json_response({"response": response})

	from agents.intent_classifier import run_intent_classifier

	response = await run_intent_classifier(query)
	return web.json_response({"response": response})


async def handle_query_ws(request: web.Request) -> web.WebSocketResponse:
	"""WebSocket text query endpoint.

	Client sends: {"query": "..."}
	Server pushes:
	  {"stage": "status",     "message": "Checking weather"}   — intent resolved
	  {"stage": "text_chunk", "text":    "..."}                 — streaming token
	  {"stage": "done",       "text":    "..."}                 — final answer
	  {"stage": "error",      "error":   "..."}                 — failure
	"""
	ws = web.WebSocketResponse()
	await ws.prepare(request)

	try:
		msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
	except asyncio.TimeoutError:
		await ws.close(code=1001, message=b"no query received within 10s")
		return ws

	if msg.type != WSMsgType.TEXT:
		await ws.send_json({"stage": "error", "error": "expected JSON text frame"})
		await ws.close()
		return ws

	try:
		data = json.loads(msg.data)
		query = data.get("query", "").strip()
	except (json.JSONDecodeError, AttributeError):
		await ws.send_json({"stage": "error", "error": "invalid JSON"})
		await ws.close()
		return ws

	if not query:
		await ws.send_json({"stage": "error", "error": "query is required"})
		await ws.close()
		return ws

	loop = asyncio.get_running_loop()
	queue: asyncio.Queue = asyncio.Queue()

	def on_status(message: str) -> None:
		queue.put_nowait({"stage": "status", "message": message})

	def on_token(token: str) -> None:
		queue.put_nowait({"stage": "text_chunk", "text": token})

	async def run_query() -> None:
		try:
			from agents.intent_classifier import run_intent_classifier

			if query.startswith("/planner"):
				task = query[len("/planner") :].strip()
				if not task:
					queue.put_nowait({"stage": "error", "error": "task is required"})
					return
				on_status("Building your plan")
				from skills.planner import run_planner

				response = await run_planner(task)
			else:
				response = await run_intent_classifier(
					query, on_token=on_token, on_status=on_status
				)
			queue.put_nowait({"stage": "done", "text": response})
		except Exception as e:
			logger.error(f"[api/query/ws] pipeline failed: {e}")
			queue.put_nowait({"stage": "error", "error": "Something went wrong. Please try again."})

	async def pump_events() -> None:
		while True:
			event = await queue.get()
			if not ws.closed:
				await ws.send_json(event)
			if event.get("stage") in ("done", "error"):
				return

	async def watch_disconnect() -> None:
		async for ws_msg in ws:
			if ws_msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR, WSMsgType.CLOSED):
				break

	query_task = asyncio.create_task(run_query())
	pump_task = asyncio.create_task(pump_events())
	watch_task = asyncio.create_task(watch_disconnect())

	_, pending = await asyncio.wait(
		[pump_task, watch_task], return_when=asyncio.FIRST_COMPLETED
	)
	for task in pending:
		task.cancel()
		try:
			await task
		except asyncio.CancelledError:
			pass
	query_task.cancel()
	try:
		await query_task
	except asyncio.CancelledError:
		pass

	if not ws.closed:
		await ws.close()

	return ws


async def handle_voice(request: web.Request) -> web.Response:
	from tts.tts import active_voice_id, voice_ids
	from agents.persona import get_active_voice_id

	voice_id = (
		request.rel_url.query.get("voice_id") or get_active_voice_id() or active_voice_id
	)
	if voice_id not in voice_ids:
		return web.json_response(
			{"error": f"unknown voice_id '{voice_id}'. Available: {voice_ids}"}, status=400
		)

	reader = await request.multipart()
	field = await reader.next()
	if field is None or field.name != "audio":
		logger.warning("[api/voice] missing audio field in multipart request")
		return web.json_response({"error": "audio field is required"}, status=400)

	with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
		audio_in = tmp.name
		while chunk := await field.read_chunk(8192):
			tmp.write(chunk)

	file_size = os.path.getsize(audio_in)
	job_id = str(uuid.uuid4())
	cancel_event = threading.Event()
	logger.info(
		f"[api/voice] job {job_id} queued — file size {file_size} bytes, voice={voice_id}"
	)
	_voice_jobs[job_id] = {"status": "pending", "cancel_event": cancel_event}

	loop = asyncio.get_running_loop()
	loop.run_in_executor(
		_voice_executor, process_voice_job, _voice_jobs[job_id], audio_in, loop, voice_id
	)

	return web.json_response({"job_id": job_id, "status": "pending"}, status=202)


async def handle_voice_ws(request: web.Request) -> web.WebSocketResponse:
	from tts.tts import active_voice_id, voice_ids
	from agents.persona import get_active_voice_id

	voice_id = (
		request.rel_url.query.get("voice_id") or get_active_voice_id() or active_voice_id
	)
	ws = web.WebSocketResponse()
	await ws.prepare(request)

	if voice_id not in voice_ids:
		await ws.send_json(
			{
				"stage": "error",
				"error": f"unknown voice_id '{voice_id}'. Available: {voice_ids}",
			}
		)
		await ws.close()
		return ws

	# Expect a single binary audio frame within 10 seconds
	try:
		msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
	except asyncio.TimeoutError:
		await ws.close(code=1001, message=b"no audio received within 10s")
		return ws

	if msg.type != WSMsgType.BINARY:
		await ws.send_json({"stage": "error", "error": "expected binary audio frame"})
		await ws.close()
		return ws

	with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
		audio_in = tmp.name
		tmp.write(msg.data)

	cancel_event = threading.Event()
	loop = asyncio.get_running_loop()
	queue: asyncio.Queue = asyncio.Queue()

	def send_event(event: dict) -> None:
		loop.call_soon_threadsafe(queue.put_nowait, event)

	def on_token(token: str) -> None:
		# Called from within the event loop (inside run_intent_classifier coroutine)
		queue.put_nowait({"stage": "text_chunk", "text": token})

	loop.run_in_executor(
		_voice_executor,
		process_voice_ws,
		audio_in,
		loop,
		voice_id,
		cancel_event,
		send_event,
		on_token,
	)

	async def pump_events() -> None:
		while True:
			event = await queue.get()
			if not ws.closed:
				try:
					await ws.send_json(event)
				except Exception:
					cancel_event.set()
					return
			if event.get("stage") in ("done", "error"):
				return

	async def watch_disconnect() -> None:
		async for ws_msg in ws:
			if ws_msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR, WSMsgType.CLOSED):
				break
		cancel_event.set()

	pump_task = asyncio.create_task(pump_events())
	watch_task = asyncio.create_task(watch_disconnect())
	_, pending = await asyncio.wait(
		[pump_task, watch_task], return_when=asyncio.FIRST_COMPLETED
	)
	for task in pending:
		task.cancel()
		try:
			await task
		except asyncio.CancelledError:
			pass

	if not ws.closed:
		await ws.close()

	return ws


async def handle_voice_cancel(request: web.Request) -> web.Response:
	job_id = request.match_info["job_id"]
	job = _voice_jobs.get(job_id)
	if job is None:
		return web.json_response({"error": "job not found"}, status=404)
	if job["status"] in ("done", "cancelled", "error"):
		return web.json_response({"error": f"job already {job['status']}"}, status=409)
	job["cancel_event"].set()
	logger.info(f"[api/voice] job {job_id} cancel requested")
	return web.json_response({"job_id": job_id, "status": "cancelling"})


async def handle_voice_result(request: web.Request) -> web.Response:
	job_id = request.match_info["job_id"]
	job = _voice_jobs.get(job_id)
	if job is None:
		logger.warning(f"[api/voice] job {job_id} not found")
		return web.json_response({"error": "job not found"}, status=404)
	status = job["status"]
	if status == "pending":
		logger.debug(f"[api/voice] job {job_id} still pending")
		return web.json_response({"status": "pending"}, status=202)
	if status == "cancelled":
		result = {k: v for k, v in job.items() if k not in ("cancel_event",)}
		_voice_jobs.pop(job_id)
		return web.json_response(result, status=200)
	if status == "error":
		logger.error(f"[api/voice] job {job_id} failed: {job['error']}")
		_voice_jobs.pop(job_id)
		return web.json_response({"error": job["error"]}, status=422)
	# done — strip internal fields before returning
	result = {k: v for k, v in job.items() if k != "cancel_event"}
	logger.info(f"[api/voice] job {job_id} done")
	_voice_jobs.pop(job_id)
	return web.json_response(result)


async def handle_compact_ws(request: web.Request) -> web.WebSocketResponse:
	from datetime import date
	from db.schemas import Conversations
	from skills.journal.compact import compact_conversations

	date_str = request.rel_url.query.get("date")
	ws = web.WebSocketResponse()
	await ws.prepare(request)

	try:
		for_date = date.fromisoformat(date_str) if date_str else date.today()
	except ValueError:
		await ws.send_json(
			{"status": "error", "error": "invalid date format, use YYYY-MM-DD"}
		)
		await ws.close()
		return ws

	doc = await Conversations.find_one(Conversations.date == for_date)
	if doc is None:
		await ws.send_json(
			{"status": "error", "error": f"no conversations found for {for_date}"}
		)
		await ws.close()
		return ws
	if not doc.exchanges:
		await ws.send_json({"status": "error", "error": "no exchanges to compact"})
		await ws.close()
		return ws
	if doc.compact_status == "pending":
		await ws.send_json({"status": "error", "error": "compact already in progress"})
		await ws.close()
		return ws

	doc.compact_status = "pending"
	await doc.save()
	await ws.send_json({"status": "pending", "date": for_date.isoformat()})

	try:
		summary = await compact_conversations(for_date)
		if not ws.closed:
			await ws.send_json(
				{"status": "complete", "date": for_date.isoformat(), "compacted": summary}
			)
		from skills.journal.journal_writer import write_journal_entry

		asyncio.create_task(write_journal_entry(for_date))
	except Exception as e:
		# Reset status so the endpoint is not stuck at "pending" on any failure path.
		# compact_conversations handles this internally for LLM errors, but early
		# ValueErrors (e.g. race on doc state) bubble up without touching the DB.
		try:
			fresh = await Conversations.find_one(Conversations.date == for_date)
			if fresh and fresh.compact_status == "pending":
				fresh.compact_status = "error"
				await fresh.save()
		except Exception:
			pass
		if not ws.closed:
			await ws.send_json({"status": "error", "error": str(e)})
	finally:
		if not ws.closed:
			await ws.close()

	return ws


async def handle_conversations(request: web.Request) -> web.Response:
	from db.schemas import Conversations

	docs = await Conversations.find().sort(-Conversations.date).to_list()
	return web.json_response({"dates": [doc.date.isoformat() for doc in docs]})


async def handle_conversation_detail(request: web.Request) -> web.Response:
	from datetime import date
	from db.schemas import Conversations

	date_str = request.match_info["date"]
	try:
		for_date = date.fromisoformat(date_str)
	except ValueError:
		return web.json_response(
			{"error": "invalid date format, use YYYY-MM-DD"}, status=400
		)

	doc = await Conversations.find_one(Conversations.date == for_date)
	if doc is None:
		return web.json_response(
			{
				"date": for_date.isoformat(),
				"exchanges": [],
			}
		)

	exchanges = sorted(doc.exchanges, key=lambda ex: ex.timestamp)
	return web.json_response(
		{
			"date": for_date.isoformat(),
			"exchanges": [
				{
					"timestamp": ex.timestamp.isoformat(),
					"question": ex.question,
					"answer": ex.answer,
				}
				for ex in exchanges
			],
		}
	)


async def handle_journal_list(request: web.Request) -> web.Response:
	from datetime import date
	from db.schemas import Journal

	date_str = request.rel_url.query.get("date")
	if date_str:
		try:
			for_date = date.fromisoformat(date_str)
		except ValueError:
			return web.json_response(
				{"error": "invalid date format, use YYYY-MM-DD"}, status=400
			)
		entry = await Journal.find_one(Journal.datestamp == for_date)
		entries = [entry] if entry else []
	else:
		limit = int(request.rel_url.query.get("limit", "10"))
		entries = await Journal.find().sort(-Journal.datestamp).limit(limit).to_list()

	return web.json_response(
		{
			"entries": [
				{"datestamp": e.datestamp.isoformat(), "summary": e.summary} for e in entries
			]
		}
	)


async def handle_plans(request: web.Request) -> web.Response:
	from db.schemas import Plans

	plans = await Plans.find().sort(-Plans.created_at).to_list()
	return web.json_response(
		{
			"plans": [
				{
					"id": str(p.id),
					"ask": p.ask,
					"plan": p.plan,
					"created_at": p.created_at.isoformat(),
				}
				for p in plans
			]
		}
	)


async def handle_notes(request: web.Request) -> web.Response:
	from db.schemas import Note

	notes = await Note.find().sort(-Note.created_at).to_list()
	return web.json_response(
		{
			"notes": [
				{
					"id": str(n.id),
					"title": n.title,
					"content": n.content,
					"category": n.category,
					"created_at": n.created_at.isoformat(),
				}
				for n in notes
			]
		}
	)


async def handle_reminders(request: web.Request) -> web.Response:
	from db.schemas import Reminders

	include_completed = (
		request.rel_url.query.get("include_completed", "false").lower() == "true"
	)

	query = (
		Reminders.find()
		if include_completed
		else Reminders.find(Reminders.is_completed == False)
	)
	reminders = await query.sort(Reminders.reminder_datetime).to_list()

	return web.json_response(
		{
			"reminders": [
				{
					"id": str(r.id),
					"description": r.description,
					"reminder_datetime": r.reminder_datetime.isoformat(),
					"is_completed": r.is_completed,
				}
				for r in reminders
			]
		}
	)


async def handle_tts_voices(request: web.Request) -> web.Response:
	from tts.tts import active_voice_id, voice_ids

	return web.json_response({"voices": voice_ids, "default": active_voice_id})


async def handle_personas(request: web.Request) -> web.Response:
	from agents.persona import PERSONAS
	from tts.tts import active_voice_id

	return web.json_response(
		{
			"personas": [
				{"id": pid, "name": p["name"], "voice_id": p["voice_id"] or active_voice_id}
				for pid, p in PERSONAS.items()
			]
		}
	)


async def handle_get_persona(request: web.Request) -> web.Response:
	from agents.persona import PERSONAS, get_active_persona_id
	from tts.tts import active_voice_id

	persona_id = get_active_persona_id()
	persona = PERSONAS[persona_id]
	return web.json_response(
		{
			"persona": persona_id,
			"name": persona["name"],
			"voice_id": persona["voice_id"] or active_voice_id,
		}
	)


async def handle_set_persona(request: web.Request) -> web.Response:
	from agents.persona import PERSONAS, set_active_persona
	from tts.tts import active_voice_id, set_active_voice_id
	from db.preferences import update_preferences

	data = await request.json()
	persona_id = data.get("persona", "").strip()
	if not persona_id:
		return web.json_response({"error": "persona is required"}, status=400)
	if persona_id not in PERSONAS:
		return web.json_response(
			{"error": f"unknown persona '{persona_id}'. Available: {list(PERSONAS.keys())}"},
			status=400,
		)
	set_active_persona(persona_id)
	persona = PERSONAS[persona_id]
	resolved_voice = persona["voice_id"] or active_voice_id
	set_active_voice_id(resolved_voice)
	await update_preferences(agentPersona=persona_id, agentVoiceId=resolved_voice)
	logger.info(f"[api/persona] active persona set to '{persona_id}' voice='{resolved_voice}'")
	return web.json_response(
		{
			"persona": persona_id,
			"name": persona["name"],
			"voice_id": resolved_voice,
		}
	)


async def handle_set_voice(request: web.Request) -> web.Response:
	from tts.tts import voice_ids, set_active_voice_id
	from db.preferences import update_preferences

	data = await request.json()
	voice_id = data.get("voice_id", "").strip()
	if not voice_id:
		return web.json_response({"error": "voice_id is required"}, status=400)
	if voice_id not in voice_ids:
		return web.json_response(
			{"error": f"unknown voice '{voice_id}'. Available: {voice_ids}"},
			status=400,
		)
	set_active_voice_id(voice_id)
	await update_preferences(agentVoiceId=voice_id)
	logger.info(f"[api/voice] active voice set to '{voice_id}'")
	return web.json_response({"voice_id": voice_id})


async def handle_health(request: web.Request) -> web.Response:
	return web.json_response({"status": "ok"})


async def handle_openapi(request: web.Request) -> web.Response:
	from api.openapi import build_spec

	return web.Response(
		text=json.dumps(build_spec(), indent=2),
		content_type="application/json",
	)


async def handle_docs(request: web.Request) -> web.Response:
	html = """<!DOCTYPE html>
<html>
<head>
  <title>jeanclawd API</title>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({ url: "/openapi.json", dom_id: "#swagger-ui" })
  </script>
</body>
</html>"""
	return web.Response(text=html, content_type="text/html")


_CORS_HEADERS = {
	"Access-Control-Allow-Origin": "*",
	"Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
	"Access-Control-Allow-Headers": "Content-Type, Authorization",
}


@web.middleware
async def log_middleware(request: web.Request, handler):
	response = await handler(request)
	if request.path != "/health":
		logger.info(f"{request.method} {request.path} → {response.status}")
	return response


@web.middleware
async def cors_middleware(request: web.Request, handler):
	if request.method == "OPTIONS":
		return web.Response(headers=_CORS_HEADERS)
	response = await handler(request)
	response.headers.update(_CORS_HEADERS)
	return response


def create_app() -> web.Application:
	app = web.Application(middlewares=[log_middleware, cors_middleware])
	app.router.add_get("/health", handle_health)
	app.router.add_post("/query", handle_query)
	app.router.add_get("/query/ws", handle_query_ws)
	app.router.add_get("/tts/voices", handle_tts_voices)
	app.router.add_get("/personas", handle_personas)
	app.router.add_get("/persona", handle_get_persona)
	app.router.add_post("/persona", handle_set_persona)
	app.router.add_post("/persona/voice", handle_set_voice)
	app.router.add_get("/voice/ws", handle_voice_ws)
	app.router.add_post("/voice", handle_voice)
	app.router.add_get("/voice/{job_id}", handle_voice_result)
	app.router.add_delete("/voice/{job_id}", handle_voice_cancel)
	app.router.add_get("/compact", handle_compact_ws)
	app.router.add_get("/conversations", handle_conversations)
	app.router.add_get("/conversation/{date}", handle_conversation_detail)
	app.router.add_get("/journal", handle_journal_list)
	app.router.add_get("/plans", handle_plans)
	app.router.add_get("/notes", handle_notes)
	app.router.add_get("/reminders", handle_reminders)
	app.router.add_get("/openapi.json", handle_openapi)
	app.router.add_get("/docs", handle_docs)
	return app


async def start_api_server(host: str = "0.0.0.0", port: int = 8000) -> web.AppRunner:
	app = create_app()
	runner = web.AppRunner(app)
	await runner.setup()
	site = web.TCPSite(runner, host, port)
	await site.start()
	return runner


if __name__ == "__main__":
	import asyncio

	async def _main() -> None:
		from db.init import db_init

		await db_init()
		runner = await start_api_server()
		try:
			await asyncio.Event().wait()
		finally:
			await runner.cleanup()

	asyncio.run(_main())
