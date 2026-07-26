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

_SANITIZED_REPLACEMENTS = str.maketrans({"/": "_", "[": "_", "]": "_"})


def _sanitize_test_name(name: str) -> str:
    """Turn a test name into a safe filename component."""
    return name.translate(_SANITIZED_REPLACEMENTS)


class PhysicsTestLogger:
    """Logs physics test results with diagnostics to per-test JSON files.

    Each test gets a context manager that captures:
    - test name, parameters, status, duration
    - arbitrary diagnostic key-value pairs logged during the test
    - error message and traceback on failure

    Each completed test is immediately persisted to:
        logs/test_<sanitized_name>.json
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
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
            self._write_entry(entry)

    def _write_entry(self, entry: dict[str, Any]) -> None:
        """Persist a single test result to its own JSON file."""
        safe_name = _sanitize_test_name(entry["test_name"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"test_{safe_name}.json"
        with open(path, "w") as f:
            json.dump(entry, f, indent=2, default=str)

    def save(self) -> None:
        """No-op — entries are written individually in test()."""
        pass


@pytest.fixture(scope="session")
def phys_log():
    """Session-scoped fixture that yields a PhysicsTestLogger.

    Each test result is written immediately to logs/test_<name>.json.
    """
    log = PhysicsTestLogger(Path("logs"))
    yield log
    log.save()
