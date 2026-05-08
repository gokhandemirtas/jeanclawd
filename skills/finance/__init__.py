from skills.base import Skill
from skills.finance.skill import handle_finance_stocks

# The finance intent prompt is dynamic (tickers are injected at runtime).
# See skills/finance/prompts.py :: get_finance_intent_prompt for the full template.
finance_skill = Skill(
    id="FINANCE_STOCKS",
    description="Stock prices and financial market data (current and historical)",
    system_prompt=None,  # Dynamic: tickers injected at runtime via get_finance_intent_prompt()
    handle=handle_finance_stocks,
)
