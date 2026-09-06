# Agent engineering guidance

## Prefer semantic, context-aware reasoning

For natural-language understanding, follow-up resolution, intent routing, and
conversation memory, use the language model with the relevant conversation
context. The router must receive enough context to reason about references such
as "earlier", "our research", "the thing we discussed", and paraphrases of
those expressions.

Do not primarily solve semantic conversation failures by adding hardcoded
keywords, regexes, phrase lists, or one-off exception branches. Before adding a
lexical rule, inspect whether context was discarded, whether the wrong router
ran, or whether the model prompt lacks the necessary history. Fix that
information flow first.

Deterministic rules are appropriate only for narrow, high-confidence boundaries
such as:

- explicit commands with an anchored command prefix (for example, `deep
  research <topic>` or `search for <topic>`);
- safety-critical or irreversible actions;
- protocol syntax and unambiguous machine commands;
- inexpensive validation and fail-safe fallbacks.

Deterministic rules must not treat a word appearing anywhere in an utterance as
proof of intent. In particular, a mention of words such as `research`, `search`,
`weather`, or `note` is not by itself an instruction to invoke that skill.

When fixing a routing or memory bug:

1. Trace the complete request path and identify which context the classifier
   actually receives.
2. Preserve or reconstruct bounded session context before changing intent
   classification.
3. Let the context-aware router decide among conversational interpretations.
4. Add regression tests with multiple natural paraphrases, including both the
   conversational reference and the explicit command that must remain distinct.
5. Only add a deterministic guard when the boundary is genuinely explicit or
   safety-critical, and document why the model cannot safely own that decision.

Do not expand a growing exception list as a substitute for session memory,
research-report metadata, or a better routing prompt.
