from skills.base import Skill
from skills.reminder.prompts import reminderIntentPrompt
from skills.reminder.skill import handle_reminder

reminder_skill = Skill(
    id="REMINDER",
    description="Timers, reminders, alarms, and viewing upcoming schedule",
    system_prompt=reminderIntentPrompt,
    handle=handle_reminder,
    # The handler returns user-facing confirmations and clarification questions.
    # Rewriting them through CHAT can falsely imply that an incomplete reminder
    # was created.
    needs_wrapping=False,
)
