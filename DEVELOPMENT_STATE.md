# Development state

## Objective

Replace brittle keyword-first routing with context-aware intent routing. The
assistant should distinguish a new explicit research command from a natural
language reference to research or conversation that already happened in the
same session.

Primary reproduction:

```text
I want to discuss about heavy metal music
we did some research earlier didn't we
tell me about the research we did in this topic
```

The final turn must not start deep research merely because it contains the word
`research`.

## Branch

`feature/langgraph-utilization`

## Design direction

- The LLM router is the primary semantic decision-maker.
- The router receives bounded session context, including the rolling summary
  retained after deep research clears detailed history.
- Deterministic classification is limited to explicit, syntax-anchored commands
  such as `research heavy metal`, `deep research black metal`, and `sear: ...`.
- Natural-language mentions of `research`, `search`, `weather`, and similar
  words must not force a skill.
- Do not add phrase/regex exceptions for individual conversational paraphrases.
- Mem0-style extracted memory is not being used for this problem; preserve
  faithful session/research events instead.

## Work completed in this branch

- Added root `AGENTS.md` with the semantic-routing rule for future agents.
- Added bounded session-summary context to router history.
- Replaced the broad keyword-first classifier with
  `_explicit_command_classify()` in the main routing paths.
- Kept `_keyword_classify` as a compatibility alias for now.
- Added tests distinguishing explicit research commands from conversational
  research references.
- Added a router-level regression test proving the semantic router receives
  retained session-summary context for a research follow-up.

## Current implementation issue

The explicit-command refactor now deliberately delegates typoed commands such
as `deep resarch ...` and `wether in London` to the context-aware router. The
old fuzzy keyword tests now assert that behavior.

## Validation so far

- Focused routing/context suite: 19 tests passed.
- Full unittest discovery: 9 tests passed.
- Modified modules compile successfully.
- Ruff passes for all modified routing/context files. `main.py` still has 14
  pre-existing E402 import-order warnings when checked directly.

## Next steps

1. Run the application manually through the three-turn heavy-metal scenario.
2. Add or adjust persisted research-event state if the manual test shows that
   the bounded summary is insufficient for several completed reports.
3. Verify both CLI and alternate intent-classifier entry points use the same
   context-aware routing boundary.
4. Run focused tests, full relevant tests, compilation, and Ruff.
5. Update this file with final validation and any known limitations.
