from dotenv import load_dotenv

from agents.intent_classifier.intent_classifier import test
from utils.boot_info import boot

load_dotenv()

if __name__ == "__main__":
    boot()
    test()
