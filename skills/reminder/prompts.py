reminderIntentPrompt = """Extract reminder or timer information from the user's query and populate JSON with three fields:

1. TYPE: Determine the query type:
   - "LIST": User wants to see/view/list their reminders for today
   - "REMINDER": User wants to create a reminder for future dates
   - "TIMER": User wants to start a timer for immediate/short countdowns

2. DATETIME: Extract ONLY the date/time expression from the query. Use fuzzy-date format. Leave EMPTY if not mentioned or if LIST type.
   - For REMINDER/TIMER: Extract when the reminder should trigger
   - For LIST: Leave EMPTY (listing only shows today's reminders)
   - DO NOT convert to ISO format - keep the natural language expression
   - Examples: "tomorrow", "tomorrow 5pm", "next monday 3pm", "in two weeks", "in 3 days"

3. DESCRIPTION: Extract what the user wants to be reminded about. Leave EMPTY if not specified or if LIST type.
   - Extract the action or event: "call mom", "dentist appointment", "moms birthday", "submit report"
   - For LIST queries: Always leave EMPTY

JSON format: {"type": "LIST" or "REMINDER" or "TIMER", "datetime": "fuzzy-date string or empty", "description": "reminder purpose or empty"}

Examples - Creating reminders:
- "Remind me tomorrow to call mom" → {"type": "REMINDER", "datetime": "tomorrow", "description": "call mom"}
- "Set a reminder in two weeks" → {"type": "REMINDER", "datetime": "in two weeks", "description": ""}
- "Remind me to call mom" → {"type": "REMINDER", "datetime": "", "description": "call mom"}
- "Remind me next monday at 3pm to submit report" → {"type": "REMINDER", "datetime": "next monday 3pm", "description": "submit report"}

Examples - Creating timers:
- "Set a timer for 5 minutes for book club" → {"type": "TIMER", "datetime": "5 minutes", "description": "book club"}
- "Set a timer in 10 seconds for cake" → {"type": "TIMER", "datetime": "15 seconds", "description": "dentist"}

Examples - Listing reminders:
- "List my reminders" → {"type": "LIST", "datetime": "", "description": ""}
- "What are my reminders today" → {"type": "LIST", "datetime": "", "description": ""}
- "Show me my schedule" → {"type": "LIST", "datetime": "", "description": ""}
- "My calendar today" → {"type": "LIST", "datetime": "", "description": ""}
"""
