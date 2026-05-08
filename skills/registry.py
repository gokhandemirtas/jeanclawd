"""Skill registry — add new skills here to make them available to the router."""

from skills.base import Skill
from skills.conversation.skill import handle_chat
from skills.finance import finance_skill
from skills.home_control import home_control_skill
from skills.information import information_skill
from skills.newsfeed import newsfeed_skill
from skills.notes import notes_skill
from skills.reminder import reminder_skill
from skills.search import search_skill
from skills.spotify import spotify_skill
from skills.transportation import transportation_skill
from skills.weather import weather_skill

# CHAT is handled directly by the conversation skill (no intermediate handler,
# no post-processing wrap needed — handle_chat already produces final output).
chat_skill = Skill(
    id="CHAT",
    description="General conversation, greetings, corrections, and anything that doesn't fit another category",
    system_prompt=None,
    handle=handle_chat,
    needs_wrapping=False,
)

SKILLS: list[Skill] = [
    weather_skill,
    finance_skill,
    home_control_skill,
    information_skill,
    reminder_skill,
    newsfeed_skill,
    transportation_skill,
    notes_skill,
    search_skill,
    spotify_skill,
    chat_skill,
]

SKILL_MAP: dict[str, Skill] = {skill.id: skill for skill in SKILLS}
