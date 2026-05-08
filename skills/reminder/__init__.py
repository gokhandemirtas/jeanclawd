from skills.base import Skill
from skills.reminder.prompts import reminderIntentPrompt
from skills.reminder.skill import handle_reminder

reminder_skill = Skill(
    id="REMINDER",
    description="Timers, reminders, alarms, and viewing upcoming schedule",
    system_prompt=reminderIntentPrompt,
    handle=handle_reminder,
)
