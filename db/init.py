import asyncio
import os

from beanie import init_beanie
from utils.log import logger
from pymongo import AsyncMongoClient

from db.schemas import (
  Conversations,
  Credentials,
  HueConfiguration,
  Journal,
  Newsfeed,
  Note,
  Plans,
  Reminders,
  ServiceCredentials,
  UserPersona,
  Weather,
)

collections = [Conversations, Credentials, HueConfiguration, Journal, Newsfeed, Note, Plans, Reminders, ServiceCredentials, UserPersona, Weather]


async def db_init() -> str | None:
  try:
    client = AsyncMongoClient(
      f"""mongodb://{os.environ["MONGODB_USER"]}:{os.environ["MONGODB_PASSWORD"]}@{os.environ["MONGODB_HOST"]}:{os.environ["MONGODB_PORT"]}""",
      timeoutMS=5000,
    )

    await init_beanie(database=client.langbox, document_models=collections)
    return f"Initiated MongoDB with: {[col.__name__ for col in collections]}"
  except Exception as error:
    logger.error(error)
    exit("Mongo DB failed to start")


if __name__ == "__main__":
  asyncio.run(db_init())
