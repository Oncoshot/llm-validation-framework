# AGENTS.md — llm-validation-framework

Agent entry point for this repo. Read before making changes. The working tree is the
source of truth over stale docs.

> **This is a public repository** (published to PyPI as `llmvalidate`). Everything here —
> commits, branch names, PR titles, issues — is visible to the world. Do **not** reference internal tooling, customer names, or any non-public information in commit messages, code comments, or docs. However, you can mention internal tickets.

## What this is

`llmvalidate` — a standalone PyPI package that evaluates LLM-extracted structured data
against ground truth. Source in `src/llmvalidate/`, tests in `tests/`.

## Releases are automatic — your commit message controls the version

This repo uses **python-semantic-release**. Every push to `master` (via merged PR) runs
`.github/workflows/ci-release.yml`, which reads the **Conventional Commit** history since
the last release, bumps the version (`src/llmvalidate/__init__.py` + `pyproject.toml`),
tags it, and **publishes to PyPI**. There is no manual version bump — never edit
`__version__` by hand.

The prefix exists **only to trigger a release** — it is not a style rule for every commit.
semantic-release scans all commits merged to `master` and bumps the version from the
highest-ranked type it finds (`feat:` > `fix:`). So **one commit per PR carrying the right
prefix is enough**; the rest can have any message you like. If no commit in the PR has a
recognized type, **no release fires**. When a prefix is used, the type token must come first,
before the colon.

### Commit format (for the release-triggering commit)

At least one commit in the PR should use:

```
<type>: <imperative description>
```

| Type            | Effect on version | Use for                          |
|-----------------|-------------------|----------------------------------|
| `fix:`          | patch (0.0.x)     | bug fixes                        |
| `feat:`         | minor (0.x.0)     | new features                     |
| `BREAKING CHANGE:` footer | major (x.0.0) | backward-incompatible changes |
| `docs:`, `chore:`, `test:`, `refactor:` | no release | non-shipping changes |

Examples:
- `fix: correct all-zero metrics for binary fields`
- `feat: add per-field confidence scoring`