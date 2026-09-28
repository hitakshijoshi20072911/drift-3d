"""DRIFTX local API: demos, run manifests, uploads, and benchmark processing."""
from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from driftx.benchmark import run_benchmark
from driftx.models import detect_model_capabilities

ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOTS = (ROOT / "outputs", ROOT / "frontend" / "demo")
RUN_ROOT = Path(os.environ.get("DRIFTX_RUN_ROOT", ROOT / "outputs" / "api_runs")).expanduser().resolve()
DEMO_MANIFEST_PATH = ROOT / "frontend" / "demo_manifest.json"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = int(os.environ.get("DRIFTX_MAX_UPLOAD_BYTES", str(512 * 1024 * 1024)))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_demo_manifest() -> list[dict[str, Any]]:
    if not DEMO_MANIFEST_PATH.is_file():
        raise RuntimeError(f"Missing demo manifest: {DEMO_MANIFEST_PATH}")
    payload = json.loads(DEMO_MANIFEST_PATH.read_text(encoding="utf-8"))
    demos = payload.get("demos", payload if isinstance(payload, list) else [])
    if len(demos) != 3:
        raise RuntimeError(f"Expected exactly 3 demos, found {len(demos)}")
    return demos


def _demo_record(demo: dict[str, Any], request: Request | None = None) -> dict[str, Any]:
    demo_id = str(demo["demo_id"])
    relative_root = Path(str(demo["artifact_root"]))
    root = (ROOT / relative_root).resolve()
    if not root.is_dir() or not any(root.is_relative_to(allowed.resolve()) for allowed in DEMO_ROOTS):
        raise RuntimeError(f"Demo artifact root is invalid: {relative_root}")
    base = f"/api/demos/{demo_id}/artifacts/"
    record = dict(demo)
    record.update({
        "run_id": f"demo-{demo_id}",
        "status": "precomputed",
        "processing_mode": "precomputed",
        "representation": "mesh/glb" if (root / "scene.glb").is_file() or (root / "model.glb").is_file() else "not measured",
        "artifact_base_url": base,
        "artifacts": {
            name: (base + value if value and name != "depth_dir" else None) for name, value in demo.get("artifacts", {}).items()
        },
        "depth_frame_count": len(list((root / "depth_vis").glob("*.jpg"))) if (root / "depth_vis").is_dir() else 0,
        "quality": demo.get("quality", "Not validated"),
        "georeference": demo.get("georeference", "Not measured"),
    })
    return record


def _safe_demo_file(demo: dict[str, Any], filename: str) -> Path:
    root = (ROOT / str(demo["artifact_root"])).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=404, detail="Demo artifact not found")
    return path


def _manifest_from_report(run_id: str, run_dir: Path, report: dict[str, Any], requested: dict[str, Any]) -> dict[str, Any]:
    artifacts = report.get("artifacts", {}) if isinstance(report.get("artifacts"), dict) else {}
    representation = "gaussian" if report.get("gs_enabled") and report.get("status") == "completed" else (
        "mesh/glb" if artifacts.get("glb") else "point-cloud" if artifacts.get("ply") else "not measured"
    )
    artifact_urls = {}
    for key, value in artifacts.items():
        if not value:
            continue
        try:
            relative = Path(str(value)).resolve().relative_to(run_dir.resolve())
        except ValueError:
            relative = Path(str(value)).name
        artifact_urls[key] = f"/api/runs/{run_id}/artifacts/{str(relative).replace(os.sep, '/') }"
    return {
        "run_id": run_id,
        "name": requested.get("name", run_id),
        "source_video": report.get("video", "not measured"),
        "model": report.get("model_variant", requested.get("model", "not measured")),
        "model_capabilities": report.get("model_capabilities"),
        "processing_mode": requested.get("reconstruction_mode", "baseline"),
        "profile": requested.get("profile", "smoke"),
        "representation": representation,
        "status": report.get("status", "not measured"),
        "duration": report.get("video_duration_s", "not measured"),
        "source_fps": report.get("source_fps", "not measured"),
        "processed_frames": report.get("frames_processed", "not measured"),
        "processing_time": report.get("total_runtime_seconds", "not measured"),
        "inference_time": report.get("inference_seconds", "not measured"),
        "peak_vram": report.get("peak_gpu_memory_mb", "not measured"),
        "confidence": report.get("mean_confidence", "Not measured"),
        "coverage": report.get("valid_depth_percentage", "Not measured"),
        "georeference": "Not measured",
        "quality": "Not validated",
        "error": report.get("error") or report.get("gaussian_fallback_error"),
        "artifacts": artifact_urls,
        "created_at": requested.get("created_at", _now()),
        "fallback": {
            "attempted": requested.get("reconstruction_mode") in {"gaussian", "both"},
            "message": "Gaussian representation unavailable for this run. Showing mesh/point-cloud reconstruction." if representation != "gaussian" and requested.get("reconstruction_mode") in {"gaussian", "both"} else None,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


app = FastAPI(title="DRIFTX API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.environ.get("DRIFTX_ALLOWED_ORIGINS", "*").split(",")],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "driftx-api", "version": app.version, "run_root": str(RUN_ROOT)}


@app.get("/api/demos")
def demos() -> dict[str, Any]:
    return {"demos": [_demo_record(demo) for demo in _load_demo_manifest()]}


@app.get("/api/demos/{demo_id}")
def demo_detail(demo_id: str) -> dict[str, Any]:
    for demo in _load_demo_manifest():
        if demo["demo_id"] == demo_id:
            return _demo_record(demo)
    raise HTTPException(status_code=404, detail="Unknown demo_id")


@app.api_route("/api/demos/{demo_id}/artifacts/{filename:path}", methods=["GET", "HEAD"])
def demo_artifact(demo_id: str, filename: str):
    for demo in _load_demo_manifest():
        if demo["demo_id"] == demo_id:
            return FileResponse(_safe_demo_file(demo, filename))
    raise HTTPException(status_code=404, detail="Unknown demo_id")


@app.post("/api/runs/upload")
async def upload_run(request: Request) -> JSONResponse:
    filename = request.headers.get("x-filename", "upload.mp4")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported video extension {suffix or '(missing)'}")
    declared_length = request.headers.get("content-length")
    if declared_length:
        try:
            if int(declared_length) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Uploaded video exceeds the configured size limit")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header") from exc
    run_id = uuid.uuid4().hex[:12]
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    video_path = run_dir / f"input{suffix}"
    received = 0
    try:
        with video_path.open("wb") as output:
            async for chunk in request.stream():
                received += len(chunk)
                if received > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded video exceeds the configured size limit")
                output.write(chunk)
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise
    if received == 0:
        shutil.rmtree(run_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded video is empty")
    state = {"run_id": run_id, "status": "uploaded", "input": str(video_path), "created_at": _now()}
    _write_json(run_dir / "job.json", state)
    with _LOCK:
        _JOBS[run_id] = state
    return JSONResponse({"run_id": run_id, "status": "uploaded"}, status_code=201)


def _process_job(run_id: str, options: dict[str, Any]) -> None:
    run_dir = RUN_ROOT / run_id
    state = {"run_id": run_id, "status": "processing", "created_at": options.get("created_at", _now())}
    with _LOCK:
        _JOBS[run_id] = state
    _write_json(run_dir / "job.json", state)
    model = str(options.get("model") or os.environ.get("DRIFTX_MODEL", "depth-anything/DA3-LARGE-1.1"))
    requested_mode = str(options.get("reconstruction_mode", "baseline"))
    try:
        report = run_benchmark(
            video=options["video"], output=run_dir, model=model,
            device=str(options.get("device", "auto")), profile=str(options.get("profile", "smoke")),
            reconstruction_mode=requested_mode,
        )
        if requested_mode in {"gaussian", "both"} and report.get("status") != "completed":
            fallback = run_benchmark(
                video=options["video"], output=run_dir / "fallback_baseline", model=model,
                device=str(options.get("device", "auto")), profile=str(options.get("profile", "smoke")),
                reconstruction_mode="baseline",
            )
            gaussian_report = report
            if fallback.get("status") == "completed":
                report = fallback
                report["gaussian_fallback_error"] = gaussian_report.get("error", "Gaussian reconstruction was unavailable")
            else:
                report["fallback_report"] = fallback
        manifest = _manifest_from_report(run_id, run_dir, report, options)
        _write_json(run_dir / "manifest.json", manifest)
        state = {"run_id": run_id, "status": report.get("status", "not measured"), "manifest": manifest, "error": report.get("error")}
    except Exception as exc:
        state = {"run_id": run_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        _write_json(run_dir / "manifest.json", {
            "run_id": run_id, "status": "failed", "representation": "not measured",
            "error": state["error"], "quality": "Not validated", "artifacts": {}, "created_at": state["created_at"],
        })
    _write_json(run_dir / "job.json", state)
    with _LOCK:
        _JOBS[run_id] = state


@app.post("/api/runs/{run_id}/process")
async def process_run(run_id: str, request: Request) -> JSONResponse:
    run_dir = RUN_ROOT / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Unknown run_id")
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    job = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
    if job.get("status") in {"processing", "completed", "failed"}:
        return JSONResponse({"run_id": run_id, "status": job["status"]}, status_code=200)
    options = {**job, **body, "video": job["input"], "created_at": job.get("created_at", _now())}
    thread = threading.Thread(target=_process_job, args=(run_id, options), daemon=True)
    thread.start()
    return JSONResponse({"run_id": run_id, "status": "processing"}, status_code=202)


def _read_job(run_id: str) -> dict[str, Any]:
    path = RUN_ROOT / run_id / "job.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown run_id")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    job = _read_job(run_id)
    manifest = RUN_ROOT / run_id / "manifest.json"
    return {"job": job, "manifest": json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else None}


@app.get("/api/runs/{run_id}/status")
def run_status(run_id: str) -> dict[str, Any]:
    job = _read_job(run_id)
    return {"run_id": run_id, "status": job.get("status", "not measured"), "error": job.get("error")}


@app.api_route("/api/runs/{run_id}/artifacts/{filename:path}", methods=["GET", "HEAD"])
def run_artifact(run_id: str, filename: str):
    root = (RUN_ROOT / run_id).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=404, detail="Run artifact not found")
    return FileResponse(path)


@app.get("/api/runs/{run_id}/manifest")
def run_manifest(run_id: str) -> dict[str, Any]:
    path = RUN_ROOT / run_id / "manifest.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Run manifest not ready")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("driftx.server:app", host=os.environ.get("DRIFTX_HOST", "127.0.0.1"), port=int(os.environ.get("DRIFTX_PORT", "8123")), reload=False)
