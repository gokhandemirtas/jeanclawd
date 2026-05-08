from skills.base import Skill
from skills.search.skill import handle_search

search_skill = Skill(
    id="SEARCH",
    description='Google web search — triggered by "search: [topic]"',
    system_prompt=None,
    handle=handle_search,
)
