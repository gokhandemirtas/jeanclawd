import asyncio
import os

from beanie import init_beanie
from pymongo import AsyncMongoClient

from db.schemas import (
  Credentials,
  HueConfiguration,
  Interaction,
  MemoryFact,
  MemoryWorkItem,
  Newsfeed,
  Note,
  Plans,
  Reminders,
  Session,
  ServiceCredentials,
  UserSettings,
  UserProfile,
  Weather,
)

collections = [Credentials, HueConfiguration, Interaction, MemoryFact, MemoryWorkItem, Newsfeed, Note, Plans, Reminders, Session, ServiceCredentials, UserProfile, UserSettings, Weather]


async def db_init() -> str | None:
  try:
    client = AsyncMongoClient(
      f"""mongodb://{os.environ["MONGODB_USER"]}:{os.environ["MONGODB_PASSWORD"]}@{os.environ["MONGODB_HOST"]}:{os.environ["MONGODB_PORT"]}""",
      timeoutMS=5000,
    )

    await init_beanie(database=client.langbox, document_models=collections)

    # UserProfile is a deliberately small startup read model.  Ensure the
    # default document exists even before the first identity fact is extracted;
    # MongoDB otherwise does not create the collection until the first insert.
    profile = await UserProfile.find_one(UserProfile.user_id == "default")
    if profile is None:
      profile = UserProfile(user_id="default")
      identity_facts = await MemoryFact.find(
        MemoryFact.user_id == "default",
        MemoryFact.namespace == "identity",
        MemoryFact.status == "active",
      ).to_list()
      for fact in identity_facts:
        if fact.key == "name":
          profile.display_name = fact.value
        elif fact.key == "birth_date":
          profile.birth_date = fact.value
        elif fact.key == "gender":
          profile.gender = fact.value
        elif fact.key in {"location", "home_location", "city"}:
          profile.location = fact.value
      await profile.insert()
    return f"Initiated MongoDB with: {[col.__name__ for col in collections]}"
  except Exception as error:
    raise RuntimeError("MongoDB failed to start") from error


if __name__ == "__main__":
  asyncio.run(db_init())
