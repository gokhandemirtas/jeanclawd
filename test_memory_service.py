from utils.memory_service import _durable_fact_value, _grounded_in_user_message, _usable_fact_value


def test_rejects_pronoun_and_greeting_values():
	assert not _usable_fact_value("you")
	assert not _usable_fact_value("hello")
	assert not _durable_fact_value("you", namespace="preference", key="likes")


def test_requires_phrases_for_interest_and_context_noise():
	assert not _durable_fact_value("online", namespace="interest", key="work_on")
	assert not _durable_fact_value("training", namespace="context", key="topic")
	assert _durable_fact_value("online sources", namespace="interest", key="work_on")


def test_allows_meaningful_single_word_identity_and_preference_values():
	assert _durable_fact_value("Gokhan", namespace="identity", key="name")
	assert _durable_fact_value("UK", namespace="identity", key="location")
	assert _durable_fact_value("jazz", namespace="preference", key="likes")


def test_grounding_uses_cleaned_value_and_rejects_second_person_noise():
	assert _grounded_in_user_message(" UK ", "I live in UK.")
	assert not _grounded_in_user_message("you", "I like you")
