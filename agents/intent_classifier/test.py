import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
from dotenv import load_dotenv
load_dotenv()

import lmstudio as lms

from agents.intent_classifier.intent_classifier import IntentResponse
from agents.intent_classifier.prompt import INTENT_PROMPT

model_key: str = os.getenv("MODEL_GENERALIST", "")
from utils import logger

TEST_CASES = [
	# HOME_CONTROL
	("turn on the lights", "HOME_CONTROL"),
	("office lights off", "HOME_CONTROL"),
	("dim the kitchen lights", "HOME_CONTROL"),
	("lights on", "HOME_CONTROL"),
	# WEATHER
	("will it rain today", "WEATHER"),
	("london weather", "WEATHER"),
	("weather forecast for tomorrow", "WEATHER"),
	# FINANCE_STOCKS
	("check Tesla stock", "FINANCE_STOCKS"),
	("Bitcoin price", "FINANCE_STOCKS"),
	# TRANSPORTATION
	("directions to the airport", "TRANSPORTATION"),
	("how do I get from Putney to Chelsea", "TRANSPORTATION"),
	# REMINDER
	("set a timer for 10 minutes", "REMINDER"),
	("remind me to call mom at 3pm", "REMINDER"),
	("my calendar today", "REMINDER"),
	# NEWSFEED
	("news today", "NEWSFEED"),
	("what's the latest news", "NEWSFEED"),
	("give me today's headlines", "NEWSFEED"),
	# INFORMATION_QUERY
	("find out what photosynthesis is", "INFORMATION_QUERY"),
	("what is quantum computing", "INFORMATION_QUERY"),
	("who is Marie Curie", "INFORMATION_QUERY"),
	# NOTES
	("save a note about Dune", "NOTES"),
	("show my notes", "NOTES"),
	("take a note for me", "NOTES"),
	("read my note on Dune", "NOTES"),
	("take a note to cook italian for my friends", "NOTES"),
	("note: I will cook tiramisu for my friends", "NOTES"),
	("jot down: buy milk", "NOTES"),
	# SEARCH
	("search hellraiser", "SEARCH"),
	("look up Blade Runner", "SEARCH"),
	("hellraiser", "SEARCH"),
	("google best pizza in London", "SEARCH"),
	("best pizza in london", "SEARCH"),
	# SPOTIFY
	("play Bohemian Rhapsody", "SPOTIFY"),
	("pause the music", "SPOTIFY"),
	("skip this song", "SPOTIFY"),
	("what's playing on Spotify", "SPOTIFY"),
	("play some music", "SPOTIFY"),
	# PLANNER
	("plan a 2 week Japan itinerary", "PLANNER"),
	("come up with a travel plan for my Tokyo trip", "PLANNER"),
	("create a study plan for learning Japanese", "PLANNER"),
	# CHAT
	("hello", "CHAT"),
	("banana elephant purple", "CHAT"),
	("fry an egg for me", "CHAT"),
	("write me a poem about autumn", "CHAT"),
	("make me a list of movie recommendations", "CHAT"),
	("I need more friends", "CHAT"),
	("I don't like chimps", "CHAT"),
	("drive me to the airport", "CHAT"),
	("you were wrong about that", "CHAT"),
	("are we all going to die", "CHAT"),
	("will humanity survive", "CHAT"),
	("open the door", "CHAT"),
	("make me a coffee", "CHAT"),
	("I am ozymandias", "CHAT"),
]

def classify(model: lms.LLM, query: str) -> str:
	chat = lms.Chat(INTENT_PROMPT)
	chat.add_user_message(query)
	raw = ""
	for fragment in model.respond_stream(chat, response_format=IntentResponse):
		raw += fragment.content
	return IntentResponse.model_validate_json(raw).intent


def run():
	model = lms.llm(
		model_key,
		config={"gpu_offload": "max", "temperature": 0, "max_tokens": -1},
	)

	passed = 0
	failed = 0
	start = time.time()

	for query, expected in TEST_CASES:
		actual = classify(model, query)
		ok = actual == expected

		if ok:
			passed += 1
		else:
			failed += 1

		logger.result(query, expected, actual, ok, "")

	elapsed = time.time() - start
	total = len(TEST_CASES)
	logger.summary(passed, total, failed, elapsed)


if __name__ == "__main__":
	run()
