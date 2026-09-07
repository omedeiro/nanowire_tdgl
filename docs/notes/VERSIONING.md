# Versioning and changelogs

The short version: write [Conventional Commits](https://www.conventionalcommits.org/),
never touch a version number or a changelog by hand, and read the section on
numerical behaviour before deciding whether a change is breaking.

## How a release happens

[release-please](https://github.com/googleapis/release-please) watches `main`.
It parses the Conventional Commit messages since the last tag, decides the
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) bump for each package
independently, and keeps an open pull request carrying the version bumps and the
changelog entries. Merging that PR tags the release. State lives in
`release-please-config.json` and `.release-please-manifest.json`.

This means the *commit message is the input to the release*, and it is the only
input you control. A change committed as `chore:` is invisible in the changelog
and produces no version bump, however significant it was.

## Commit type → changelog section

The changelog sections are [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
headings, configured in `release-please-config.json`:

| Commit type | Section | Version bump |
|---|---|---|
| `feat:` | **Added** | minor |
| `fix:` | **Fixed** | patch |
| `perf:`, `refactor:` | **Changed** | patch |
| `deprecate:` | **Deprecated** | patch |
| `remove:`, `revert:` | **Removed** | patch |
| `security:` | **Security** | patch |
| `docs:`, `test:`, `ci:`, `build:`, `chore:` | hidden | none |
| any type with `!`, or a `BREAKING CHANGE:` footer | **⚠ BREAKING CHANGES** | major |

Scope by package: `feat(server):`, `fix(solver):`, `test(physics):`.

Keep a Changelog's remaining rule — that entries are written for humans, not
machines — is a rule about commit subjects here. `fix(solver): correct sign`
tells a reader nothing; `fix(solver): use D = ∇ − iA in the LPSI operators` tells
them whether it affects their results. The subject line *is* the changelog entry.

## What counts as breaking, in a package whose output is numbers

The public API is not the whole contract. Someone depending on `tdgl3d` depends
on the numbers it returns, so the rule this project uses is:

**Breaking (major).**

- Removing or renaming a public name, or changing a signature incompatibly.
- Changing the meaning of a parameter, a unit, or a sign convention.
- Changing a saved-file or POM schema so old files no longer load.
- **Changing a physically correct result to a different physically correct
  result** — a different discretisation, a different default that moves the
  answer, a changed convention for where a quantity is sampled. A user's
  published figure will not reproduce, and no amount of "it is more accurate
  now" makes that a patch.

**Not breaking (patch), even though the numbers move.**

- Fixing a result that was *wrong* — a gauge inconsistency, a stride bug, a
  double-counted boundary term. The previous output was not a contract; it was a
  defect. Say so explicitly in the commit body, and say what changes
  observably, so a reader can tell whether their results are affected.
- Tightening a test tolerance, or improving the accuracy of a *reference model*
  used only in tests.

**Added (minor).**

- New public functions, new solver options that default to the existing
  behaviour, new diagnostics.

When a fix falls near the line, the deciding question is whether a user could
reasonably have relied on the old behaviour. If they could, it is breaking.

## Numerical changes need a changelog entry a reader can act on

A commit that moves results should name the observable that moves and by how
much. `fix(solver): correct the Peierls phase sign` is a true subject and a
useless one; a reader wants to know that vortex windings in a uniform field were
previously mixed ±1 and are now uniformly +1. Put the magnitude in the body —
the subject stays short, and the body reaches the release notes through the
commit, not through anyone's memory.

## What not to do

- Do not edit `CHANGELOG.md` in any package. release-please owns those files and
  reconciles them against its manifest; a hand-written entry will be duplicated
  or clobbered.
- Do not edit `version` in a `pyproject.toml`, or `.release-please-manifest.json`.
- Do not tag releases by hand.
- Do not squash a breaking change into a `chore:` commit to avoid a major bump.
  The version number is a claim about compatibility; understating it is a bug
  report waiting to be filed.
