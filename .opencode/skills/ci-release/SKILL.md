---
name: ci-release
description: >
  Work on CI/CD, GitHub Actions workflows, release-please config, or
  conventional commits. Use when editing .github/, release config,
  or versioning/changelog files.
---

# CI / Release

## Conventional Commits

All commits must follow Conventional Commits format:

```
<type>(<scope>): <description>
```

- Types: `feat`, `fix`, `refactor`, `docs`, `test`, `ci`, `chore`
- Scope by package: `feat(server): ...`, `fix(solver): ...`
- Breaking changes: `!` suffix, e.g. `feat(schema)!: ...`
- release-please derives versions and changelogs from these messages.

## release-please

- Monorepo manifest mode (`.release-please-manifest.json` +
  `release-please-config.json`).
- **Never** hand-edit version numbers or CHANGELOG files — they are
  automated by release-please.
- Config lives at repo root: `release-please-config.json`.

## CI Pipeline (GitHub Actions)

- **ci.yml**: ruff lint + pytest matrix on Python 3.10 and 3.12.
- **release-please.yml**: opens/maintains release PRs on conventional commit.
- Branch protection on `main`: CI must pass before merge.

## Branching

1. Never commit directly to `main`. Create a feature branch.
2. Open a PR; CI must pass before merge.
3. Binary artifacts (GIF/PNG/HDF5/logs) are excluded via `.gitignore`.
