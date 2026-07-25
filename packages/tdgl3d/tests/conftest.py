"""Shared test fixtures and diagnostic logging for physics validation tests."""

from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pytest


class PhysicsTestLogger:
    """Logs physics test results with diagnostics to a JSON runlog.

    Each test gets a context manager that captures:
    - test name, parameters, status, duration
    - arbitrary diagnostic key-value pairs logged during the test
    - error message and traceback on failure

    Output: logs/physics_test_runlog.json
    """

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.results: list[dict[str, Any]] = []

    @contextmanager
    def test(self, name: str, params: Optional[dict[str, Any]] = None):
        """Context manager for a single test.

        Usage::

            with phys_log.test("test_name", {"Nx": 10}) as log:
                result = run_simulation(...)
                log["fitted_value"] = result.value
                assert result.value == expected
        """
        entry: dict[str, Any] = {
            "test_name": name,
            "timestamp": datetime.now().isoformat(),
            "params": params or {},
            "diagnostics": {},
            "status": "running",
            "error": None,
            "traceback": None,
            "duration_s": None,
        }
        start = time.perf_counter()
        try:
            yield entry["diagnostics"]
            entry["status"] = "passed"
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc()
            raise
        finally:
            entry["duration_s"] = round(time.perf_counter() - start, 4)
            self.results.append(entry)

    def save(self) -> None:
        """Write runlog JSON to disk."""
        summary = {
            "run_timestamp": datetime.now().isoformat(),
            "total": len(self.results),
            "passed": sum(1 for r in self.results if r["status"] == "passed"),
            "failed": sum(1 for r in self.results if r["status"] == "failed"),
            "results": self.results,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)


@pytest.fixture(scope="session")
def phys_log():
    """Session-scoped fixture that yields a PhysicsTestLogger and saves on teardown."""
    log = PhysicsTestLogger(Path("logs") / "physics_test_runlog.json")
    yield log
    log.save()
