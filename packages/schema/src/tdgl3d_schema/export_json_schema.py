"""Export the POM JSON Schema for consumption by TypeScript / other tooling.

Usage:  python3 -m tdgl3d_schema.export_json_schema [output_path]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tdgl3d_schema import Project


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pom.schema.json")
    out.write_text(json.dumps(Project.model_json_schema(), indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
