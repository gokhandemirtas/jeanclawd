from skills.base import Skill
from skills.search.skill import handle_search

_SEARCH_PROMPT = (
    "You are a voice assistant. The user asked a question and the top web search result is provided below. "
    "Answer the user's question directly and conversationally using the information in the result. "
    "Do not list options, do not say 'here are some results', do not mention URLs or sources. "
    "If the result contains the answer, state it clearly and concisely. "
    "If the result is not sufficient to answer, say so briefly."
)

search_skill = Skill(
    id="SEARCH",
    description='Google web search — triggered by "search: [topic]"',
    system_prompt=_SEARCH_PROMPT,
    handle=handle_search,
)
