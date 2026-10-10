# Issue tracker: GitHub

Issues and specs for this repo live as GitHub issues. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Create a sub-issue**: `gh issue create --title "Sub-issue Title" --body "Description" --parent PARENT-ISSUE-NUMBER`
- **Link an existing issue as a sub-issue**: `gh issue edit PARENT-ISSUE-NUMBER --add-sub-issue SUB-ISSUE-NUMBER`
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`while:

1. Ticking boxes when completing acceptance criteria before closing issue/subissue. Serves as explicit verification trail.
2. Removing ready-for-agent label when closing ticket or moving to done.

Infer the repo from `git remote -v`; `gh` does this automatically when run inside a clone.

## Pull requests as a triage surface

**PRs as a request surface: no.**

## When a skill says "publish to the issue tracker"

Create a GitHub issue. Specs map to parent GitHub issues. Implementation tickets map to GitHub subissues linked to parent issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.
