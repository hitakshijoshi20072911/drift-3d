from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


class PipelineError(RuntimeError):
    """Raised when an external reconstruction stage fails."""


def run_command(name: str, command: Sequence[str], report: dict, cwd: Path | None = None) -> str:
    """Run a command, stream its output, and record its duration and status."""
    print(f"\n=== {name} ===")
    print(" ".join(str(part) for part in command), flush=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    process = subprocess.Popen(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    lines: list[str] = []
    full_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        full_lines.append(line)
        lines.append(line)
        if len(lines) > 250:
            lines.pop(0)
    return_code = process.wait()
    elapsed = round(time.perf_counter() - started, 2)
    ended_at = datetime.now(timezone.utc).isoformat()
    log_file = None
    if report.get("_log_dir"):
        log_dir = Path(report["_log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{re.sub(r'[^a-zA-Z0-9_.-]+', '_', name)}.log"
        log_file.write_text("".join(full_lines), encoding="utf-8")
    report.setdefault("stages", {})[name] = {
        "seconds": elapsed,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "return_code": return_code,
        "command": [str(part) for part in command],
        "log_tail": "".join(lines[-80:]),
        "log_file": str(log_file) if log_file else None,
    }
    if report.get("_report_path"):
        write_json(Path(report["_report_path"]), {key: value for key, value in report.items() if not key.startswith("_")})
    if return_code:
        raise PipelineError(f"{name} failed with exit code {return_code}")
    print(f"--- {name}: {elapsed:.1f}s")
    return "".join(lines)


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
