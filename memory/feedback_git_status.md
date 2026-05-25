---
name: Always check git status before starting work
description: Run git status at the start of every task to catch uncommitted changes and avoid introducing regressions
type: feedback
---

Always run `git status` before beginning any implementation work.

**Why:** User wants to know if there are uncommitted changes that could be lost or conflated with new changes. If there are uncommitted changes, warn the user to commit first before proceeding.

**How to apply:** First tool call of every work session should be `git status`. If the tree is dirty, surface a warning and ask the user whether to proceed.
