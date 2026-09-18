# CLAUDE.md
**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing code see `docs/agents/code-preferences.md`

## Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Working Preferences

- **Be extremely concise.** Sacrifice grammar for concision.
- **Dev branch workflow.** All work happens on `dev`. Only merge to `main` with explicit user approval. Push to remote for both branches.
- **No subagent retries.** If a subagent fails, abandon it and proceed directly.
- **Confirm before host changes.** Ask before installing anything new on the host system or making system-level changes.
- **Avoid Microsoft Store** for installations whenever possible.

## Gotchas

Record gotchas encountered during development in `docs/agents/gotchas.md` so fresh sessions can avoid them.

## Agent skills

### Issue tracker

Issues tracked via GitHub Issues on this repo. See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout. See `docs/agents/domain.md`.