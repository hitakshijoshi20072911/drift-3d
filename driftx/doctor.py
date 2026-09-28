"""Environment and model preflight diagnostics for DRIFTX."""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from driftx.models import detect_model_capabilities


def _check(name: str, status: str, value: Any = None, detail: str | None = None) -> dict[str, Any]:
    item = {"name": name, "status": status}
    if value is not None:
        item["value"] = value
    if detail:
        item["detail"] = detail
    return item


def _command(name: str) -> str | None:
    path = shutil.which(name)
    if not path:
        return None
    try:
        return subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5).stdout.strip() or path
    except Exception:
        return path


def run_doctor(model: str | None = None, output: str | Path = "outputs/system/doctor_report.json") -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    checks.append(_check("python", "PASS" if sys.version_info >= (3, 11) else "WARN", platform.python_version()))
    checks.append(_check("os", "PASS", platform.platform()))
    try:
        import torch
        checks.append(_check("pytorch", "PASS", torch.__version__))
        checks.append(_check("torch_cuda_runtime", "PASS" if torch.version.cuda else "WARN", torch.version.cuda or "not available"))
        cuda = bool(torch.cuda.is_available())
        checks.append(_check("cuda_available", "PASS" if cuda else "WARN", cuda))
        if cuda:
            checks.append(_check("gpu_name", "PASS", torch.cuda.get_device_name(0)))
            checks.append(_check("gpu_vram_mb", "PASS", round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 2)))
        else:
            checks.append(_check("gpu_name", "WARN", "not available"))
            checks.append(_check("gpu_vram_mb", "WARN", "not measured"))
    except Exception as exc:
        checks.append(_check("pytorch", "WARN", detail=f"{type(exc).__name__}: {exc}; install the ML extras for inference"))
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    checks.append(_check("cuda_home", "PASS" if cuda_home else "WARN", cuda_home or "not set"))
    nvcc = shutil.which("nvcc")
    checks.append(_check("nvcc", "PASS" if nvcc else "WARN", nvcc or "not found"))
    cl = shutil.which("cl") or shutil.which("cl.exe")
    checks.append(_check("msvc", "PASS" if cl else "WARN", cl or "not found"))
    checks.append(_check("gsplat_import", "PASS" if importlib.util.find_spec("gsplat") else "WARN", bool(importlib.util.find_spec("gsplat"))))
    checks.append(_check("da3_import", "PASS" if importlib.util.find_spec("depth_anything_3") else "WARN", bool(importlib.util.find_spec("depth_anything_3"))))
    ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
    checks.append(_check("ffmpeg", "PASS" if ffmpeg else "WARN", ffmpeg or "not found"))
    for package in ("fastapi", "uvicorn", "trimesh", "imageio"):
        checks.append(_check(f"dependency:{package}", "PASS" if importlib.util.find_spec(package) else "WARN"))
    if model:
        path = Path(model).expanduser()
        checks.append(_check("model_path", "PASS" if path.exists() else "WARN", str(path), "Model path is not present in this environment" if not path.exists() else None))
        capabilities = detect_model_capabilities(model).to_dict()
    else:
        capabilities = None
    disk = shutil.disk_usage(Path.cwd())
    checks.append(_check("disk_free_gb", "PASS" if disk.free > 5 * 1024**3 else "WARN", round(disk.free / 1024**3, 2)))
    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else ("WARN" if any(c["status"] == "WARN" for c in checks) else "PASS")
    report = {"schema_version": 1, "status": status, "checks": checks, "model_capabilities": capabilities}
    report_path = Path(output).expanduser()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def print_doctor(report: dict[str, Any], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(f"DRIFTX doctor: {report['status']}")
    for check in report["checks"]:
        value = f" — {check.get('value')}" if "value" in check else ""
        detail = f" ({check['detail']})" if check.get("detail") else ""
        print(f"[{check['status']}] {check['name']}{value}{detail}")
    print("Report: outputs/system/doctor_report.json")
