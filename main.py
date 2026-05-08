import time

from dotenv import load_dotenv

from agents.greeting.greeting_agent import greet_and_collect
from agents.intent_classifier.intent_classifier import classify
from agents.intent_router.intent_router import route
from utils import logger
from utils.boot_info import boot

load_dotenv()

if __name__ == "__main__":
	boot()

	user_input = greet_and_collect()

	start = time.time()

	intent = classify(user_input)
	logger.info(f"Intent: {intent}")

	response = route(intent, user_input)

	logger.newline()
	logger.info(response)
	logger.newline()
	logger.timing(time.time() - start)
