---
name: schema-pom
description: >
  Work on POM schemas in packages/schema. Use when editing Pydantic models,
  bumping POM_VERSION, adding new POM fields, or exporting JSON Schema.
---

# POM Schema (packages/schema)

Pydantic models for the Project Object Model — the single contract shared by
the studio frontend, `tdgl3d-server`, and AI assistant tools.

## Rules

- All models forbid extra fields (`extra="forbid"`). Schema evolution happens
  by bumping `POM_VERSION` in `src/tdgl3d_schema/pom.py`, not by loosening
  validation.
- Every entity extends `POMObject` (stable `id` + `name`) so the UI tree,
  undo history, and AI tool calls can address objects individually.
- When the solver gains a feature (new BC, new material model, etc.), add it
  here AND in `packages/tdgl3d-server/src/tdgl3d_server/build.py`.
- No solver imports here — this package depends only on pydantic.

## Commands

```bash
# Export JSON Schema (used later for TypeScript codegen in the studio)
python3 -m tdgl3d_schema.export_json_schema

# Run tests
python3 -m pytest packages/schema/tests -q
```
