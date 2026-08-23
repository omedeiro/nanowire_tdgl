# Changelog

This repository is a monorepo, and each package keeps its own changelog:

| Package | Changelog | Current version |
|---|---|---|
| `tdgl3d` — the solver | [`packages/tdgl3d/CHANGELOG.md`](packages/tdgl3d/CHANGELOG.md) | see [`.release-please-manifest.json`](.release-please-manifest.json) |
| `tdgl3d-schema` — POM schemas | [`packages/schema/CHANGELOG.md`](packages/schema/CHANGELOG.md) | " |
| `tdgl3d-server` — job service | [`packages/tdgl3d-server/CHANGELOG.md`](packages/tdgl3d-server/CHANGELOG.md) | " |

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries are **generated**, not written by hand: every package changelog is
produced by [release-please](https://github.com/googleapis/release-please) from
the Conventional Commit messages on `main`, and the version numbers in
`.release-please-manifest.json` and each `pyproject.toml` are bumped by the same
tool. Editing them by hand desynchronises the tool's state from the repository.
What you can steer is the commit message — see
[`docs/notes/VERSIONING.md`](docs/notes/VERSIONING.md) for how commit types map
to Keep a Changelog sections, and for what counts as a breaking change in a
package whose output is numbers.

There is no "Unreleased" section to read here. Its equivalent is the open
release PR that release-please maintains against `main`: it accumulates every
merged commit since the last tag and shows exactly what the next changelog entry
and version bump will be.
