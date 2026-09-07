"""Shared test fixtures and diagnostic logging for physics validation tests.

The :class:`PhysicsTestLogger` fixture (``phys_log``) records what each physics
test measured, what it expected, and how much slack it allowed, so that
``docs/generate_test_report.py`` can render a report in which every row is
falsifiable.

Prefer the ``check_*`` helpers on the yielded record over bare ``assert``
statements: they force an *expected value* and a *tolerance* to be written down
next to the measurement, and both end up in the report.  A check whose tolerance
is derived from the measurement it is checking cannot fail, and is therefore not
a verification of anything.
"""

from __future__ import annotations

import inspect
import json
import math
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


def _calling_test_module() -> str:
    """Name of the test module that opened the current ``phys_log.test`` block.

    Recorded so the report can group results by the principle each suite covers
    without pattern-matching on test names.
    """
    for frame in inspect.stack():
        module = inspect.getmodule(frame.frame)
        stem = Path(frame.filename).stem
        if stem.startswith("test_"):
            return module.__name__.rsplit(".", 1)[-1] if module else stem
    return "unknown"


class PhysicsRecord(dict):
    """Diagnostics mapping for one physics test, plus assertion helpers.

    Behaves as a plain ``dict`` (``record["key"] = value`` still works, and those
    entries are reported as free-form diagnostics).  The ``check_*`` methods
    additionally append a structured entry to the test's check list:

    ``{label, measured, expected, tolerance, units, detail, passed}``

    and raise :class:`AssertionError` when the check fails.
    """

    def __init__(self, checks: list[dict[str, Any]]):
        super().__init__()
        self._checks = checks

    # -- internal ----------------------------------------------------------
    def _record(
        self,
        label: str,
        measured: float,
        expected: Any,
        tolerance: Any,
        passed: bool,
        detail: str,
        units: str = "",
    ) -> None:
        self._checks.append(
            {
                "label": label,
                "measured": float(measured),
                "expected": expected,
                "tolerance": tolerance,
                "units": units,
                "detail": detail,
                "passed": bool(passed),
            }
        )

    # -- public helpers ----------------------------------------------------
    def check_close(
        self,
        label: str,
        measured: float,
        expected: float,
        *,
        rtol: float = 0.0,
        atol: float = 0.0,
        units: str = "",
        detail: str = "",
    ) -> float:
        """Assert ``|measured - expected| <= atol + rtol*|expected|``."""
        measured = float(measured)
        expected = float(expected)
        tol = float(atol) + float(rtol) * abs(expected)
        err = abs(measured - expected)
        passed = bool(err <= tol) and math.isfinite(measured)
        self._record(
            label, measured, expected, tol, passed,
            detail or f"|measured - expected| <= {tol:.3g}", units,
        )
        if not passed:
            raise AssertionError(
                f"{label}: measured {measured:.6g}, expected {expected:.6g} "
                f"(|Δ| = {err:.3g} > tol {tol:.3g})"
                + (f" — {detail}" if detail else "")
            )
        return measured

    def check_below(
        self,
        label: str,
        measured: float,
        limit: float,
        *,
        units: str = "",
        detail: str = "",
    ) -> float:
        """Assert ``measured <= limit``."""
        measured = float(measured)
        limit = float(limit)
        passed = bool(measured <= limit) and math.isfinite(measured)
        self._record(
            label, measured, f"<= {limit:.3g}", limit, passed,
            detail or f"must not exceed {limit:.3g}", units,
        )
        if not passed:
            raise AssertionError(
                f"{label}: measured {measured:.6g}, must be <= {limit:.6g}"
                + (f" — {detail}" if detail else "")
            )
        return measured

    def check_above(
        self,
        label: str,
        measured: float,
        limit: float,
        *,
        units: str = "",
        detail: str = "",
    ) -> float:
        """Assert ``measured >= limit``."""
        measured = float(measured)
        limit = float(limit)
        passed = bool(measured >= limit) and math.isfinite(measured)
        self._record(
            label, measured, f">= {limit:.3g}", limit, passed,
            detail or f"must be at least {limit:.3g}", units,
        )
        if not passed:
            raise AssertionError(
                f"{label}: measured {measured:.6g}, must be >= {limit:.6g}"
                + (f" — {detail}" if detail else "")
            )
        return measured

    def check_within(
        self,
        label: str,
        measured: float,
        lo: float,
        hi: float,
        *,
        units: str = "",
        detail: str = "",
    ) -> float:
        """Assert ``lo <= measured <= hi``."""
        measured = float(measured)
        passed = bool(lo <= measured <= hi) and math.isfinite(measured)
        self._record(
            label, measured, f"[{lo:.3g}, {hi:.3g}]", (float(lo), float(hi)),
            passed, detail or f"must lie in [{lo:.3g}, {hi:.3g}]", units,
        )
        if not passed:
            raise AssertionError(
                f"{label}: measured {measured:.6g}, expected in [{lo:.6g}, {hi:.6g}]"
                + (f" — {detail}" if detail else "")
            )
        return measured


class PhysicsTestLogger:
    """Logs physics test results with diagnostics to per-test JSON files.

    Each test gets a context manager that captures:
    - test name, parameters, description, status, duration
    - structured checks (measured / expected / tolerance / pass)
    - arbitrary diagnostic key-value pairs logged during the test
    - error message and traceback on failure

    Each completed test is immediately persisted to:
        logs/test_<sanitized_name>.json
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.results: list[dict[str, Any]] = []

    @contextmanager
    def test(
        self,
        name: str,
        params: Optional[dict[str, Any]] = None,
        description: str = "",
    ):
        """Context manager for a single test.

        Usage::

            with phys_log.test("test_name", {"Nx": 10}, "what this proves") as log:
                result = run_simulation(...)
                log["raw_profile"] = result.profile.tolist()
                log.check_close("λ", result.lam, params.kappa, rtol=0.1)
        """
        checks: list[dict[str, Any]] = []
        entry: dict[str, Any] = {
            "test_name": name,
            "module": _calling_test_module(),
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "params": params or {},
            "checks": checks,
            "diagnostics": {},
            "status": "running",
            "error": None,
            "traceback": None,
            "duration_s": None,
        }
        record = PhysicsRecord(checks)
        start = time.perf_counter()
        try:
            yield record
            entry["status"] = "passed"
        except Exception as e:
            entry["status"] = "failed"
            entry["error"] = f"{type(e).__name__}: {e}"
            entry["traceback"] = traceback.format_exc()
            raise
        finally:
            entry["diagnostics"] = dict(record)
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
