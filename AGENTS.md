# Agent Operating Guide

This guide captures the operating rules for running a Claude Code/Fable agent team on this repository. Keep it short, explicit, and easy for an orchestra agent to follow.

## Branch Rules

- `main` is the stable branch. Do not implement directly on `main`.
- Long-running agent work happens on `agent-orchestra`.
- Implementation PRs target `agent-orchestra`, not `main`.
- The final integration PR is `agent-orchestra` to `main`.
- Use squash merge only.
- Before editing, merging, or opening a PR, run:

```sh
pwd && git branch --show-current
```

If the branch is `main`, stop and switch to the correct working branch.

## Orchestra Agent

The orchestra agent coordinates the run. It should:

- Maintain an orchestration ledger with tickets, owners, branches, PRs, checks, verdicts, and merge results.
- Deduplicate findings before assigning implementation work.
- Reject vague, low-value, duplicate, or out-of-scope findings with a short reason.
- Dispatch implementation and review agents in parallel where practical.
- Keep implementation work out of the main checkout.
- Review every implementation PR before merge.
- Require checks before merge.
- Squash merge accepted PRs into `agent-orchestra`.

The orchestra should not make product code changes itself except for explicit orchestration or emergency repair work.

## Implementation Agents

Each implementation agent should:

- Work from the latest `origin/agent-orchestra`.
- Create its own branch and worktree.
- Keep each PR tightly scoped to an accepted ticket.
- Include a PR body with problem, solution, tests run, and risk.
- Run relevant targeted tests plus the standard checks:

```sh
npm test
npm run lint
npm run build
```

- Avoid lockfile changes unless the ticket specifically requires them.
- Rebase or recreate from the latest `origin/agent-orchestra` when the base branch moves.

## Review Agents

Each PR should have a reviewing agent. Review agents should:

- Inspect the PR diff and stated acceptance criteria.
- Run or verify the relevant checks.
- Post review results as GitHub comments so the conversation is visible.
- Use PASS, ISSUES, or UNVERIFIED verdicts.
- Include expected vs observed behavior for every issue.
- Include evidence paths for logs, screenshots, probes, and command output.

When a review finds a real issue, the orchestra should create or assign a follow-up implementation ticket. The fix should land in a new PR targeting `agent-orchestra`.

## Visual Review Protocol

UI, rendering, browser, accessibility, scrolling, theming, or demo changes need real browser evidence.

Visual review agents should:

- Use their own worktree.
- Use their own local dev server port.
- Use their own browser process, context, or profile.
- Save artifacts under:

```txt
/tmp/notie-visual-review/pr-<number>/
```

- Capture screenshots, console logs, network notes when relevant, and DOM/probe outputs.
- Post a GitHub comment with setup, steps, observations, verdict, and evidence paths.

Agents should not serialize all browser work through one shared browser lock. Independent PR teams should be able to inspect frontends concurrently.

## Smoke Review Protocol

Some PRs have no UI surface, such as formatting, packaging, tests, CI, or benchmark changes. These still need a GitHub-visible review comment.

Smoke review comments should state:

- Why there is no visual surface.
- Which commands or probes were run.
- The result of each check.
- Any evidence paths.
- PASS, ISSUES, or UNVERIFIED verdict.

## GitHub Comments

GitHub should be the visible conversation record between review and implementation agents.

- Review findings go on the original PR.
- Follow-up fixes should link back to the finding that caused them.
- Final status comments should link the follow-up PR and state whether the issue was resolved.

## Completion Gate

An orchestra run is done only when:

- Every accepted ticket is merged into `agent-orchestra` or rejected with evidence.
- No PRs targeting `agent-orchestra` remain open unless explicitly documented as blocked or out of scope.
- Every merged PR has a visual review, visual re-review, or smoke review comment.
- `agent-orchestra` is synced with `origin/agent-orchestra`.
- Fresh final checks pass on `agent-orchestra`:

```sh
npm test
npm run lint
npm run build
```

- The final report summarizes merged PRs, rejected findings, follow-up fixes, remaining risks, and how to compare `agent-orchestra` with `main`.

## Practical Notes

- Prefer small, well-scoped PRs over broad bundles.
- Keep unrelated refactors out of implementation tickets.
- Treat tool backpressure as a signal to reduce concurrency, not to abandon the run.
- Keep local artifact directories out of commits unless they are intentional documentation.
- Do not delete other active Claude sessions, proxy processes, or user-owned work without explicit instruction.
