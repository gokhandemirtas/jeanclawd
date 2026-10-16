import asyncio
import os
from pathlib import Path

from beanie import init_beanie
from pymongo import AsyncMongoClient

from db.schemas import (
	Credentials,
	HueConfiguration,
	Interaction,
	IntrospectionEmbedding,
	Newsfeed,
	Note,
	Plans,
	Reminders,
	ResearchChunk,
	ResearchDocument,
	Session,
	ServiceCredentials,
	UserSettings,
	Weather,
)

collections = [
	Credentials,
	HueConfiguration,
	Interaction,
	IntrospectionEmbedding,
	Newsfeed,
	Note,
	Plans,
	Reminders,
	ResearchChunk,
	ResearchDocument,
	Session,
	ServiceCredentials,
	UserSettings,
	Weather,
]


async def db_init() -> str | None:
	try:
		client = AsyncMongoClient(
			f"""mongodb://{os.environ["MONGODB_USER"]}:{os.environ["MONGODB_PASSWORD"]}@{os.environ["MONGODB_HOST"]}:{os.environ["MONGODB_PORT"]}/?authSource=admin&directConnection=true&serverSelectionTimeoutMS=5000&connectTimeoutMS=5000""",
			timeoutMS=5000,
		)

		await init_beanie(
			database=client[os.environ.get("MONGODB_DB", "langbox")],
			document_models=collections,
		)

		# Backfill the research index for reports created before the RAG layer.
		# index_report is idempotent by report path and content hash.
		try:
			from utils.research_rag import index_report

			report_dir = Path(__file__).resolve().parents[1] / "deep-research"
			for report_path in sorted(report_dir.glob("*.md")):
				await index_report(str(report_path))
		except Exception:
			# Database startup should remain usable if an individual report cannot
			# be parsed or the optional retrieval dependency is unavailable.
			import logging

			logging.getLogger(__name__).exception("Research report backfill failed")

		return f"Initiated MongoDB with: {[col.__name__ for col in collections]}"
	except Exception as error:
		raise RuntimeError("MongoDB failed to start") from error


if __name__ == "__main__":
	asyncio.run(db_init())
