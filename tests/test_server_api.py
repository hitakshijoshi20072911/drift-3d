import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import driftx.server as server
from driftx.server import app


class ServerApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_and_exactly_three_demos(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        demos = self.client.get("/api/demos")
        self.assertEqual(demos.status_code, 200)
        payload = demos.json()["demos"]
        self.assertEqual([demo["demo_id"] for demo in payload], ["test3", "test6", "test7"])

    def test_each_demo_serves_declared_glb(self):
        for demo in self.client.get("/api/demos").json()["demos"]:
            self.assertTrue(demo["artifact_root"].startswith("outputs/"))
            self.assertEqual(demo["artifacts"]["glb"].rsplit("/", 1)[-1], "scene.glb")
            response = self.client.get(f"/api/demos/{demo['demo_id']}/artifacts/scene.glb")
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.content), 1000)
            head = self.client.head(f"/api/demos/{demo['demo_id']}/artifacts/scene.glb")
            self.assertEqual(head.status_code, 200)
            self.assertEqual(head.headers.get("content-type"), "model/gltf-binary")
            for artifact in ("scene.jpg", "depth_vis/0000.jpg"):
                evidence = self.client.get(f"/api/demos/{demo['demo_id']}/artifacts/{artifact}")
                self.assertEqual(evidence.status_code, 200)
                self.assertGreater(len(evidence.content), 1000)
            self.assertGreater(demo["depth_frame_count"], 0)

    def test_invalid_uploads_are_rejected(self):
        wrong_extension = self.client.post(
            "/api/runs/upload", headers={"X-Filename": "notes.txt"}, content=b"not a video"
        )
        self.assertEqual(wrong_extension.status_code, 415)
        empty = self.client.post(
            "/api/runs/upload", headers={"X-Filename": "empty.mp4"}, content=b""
        )
        self.assertEqual(empty.status_code, 400)

    def test_upload_size_limit_is_enforced_before_persisting(self):
        old_limit = server.MAX_UPLOAD_BYTES
        old_root = server.RUN_ROOT
        with tempfile.TemporaryDirectory() as tmp:
            server.MAX_UPLOAD_BYTES = 4
            server.RUN_ROOT = Path(tmp)
            try:
                response = self.client.post(
                    "/api/runs/upload", headers={"X-Filename": "sample.mp4"}, content=b"12345"
                )
                self.assertEqual(response.status_code, 413)
                self.assertEqual(list(Path(tmp).iterdir()), [])
            finally:
                server.MAX_UPLOAD_BYTES = old_limit
                server.RUN_ROOT = old_root

    def test_process_is_idempotent_for_terminal_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_root = server.RUN_ROOT
            server.RUN_ROOT = Path(tmp)
            try:
                run_dir = server.RUN_ROOT / "done"
                run_dir.mkdir()
                (run_dir / "job.json").write_text(
                    json.dumps({"run_id": "done", "status": "completed", "input": "input.mp4"})
                )
                response = self.client.post("/api/runs/done/process", json={"profile": "smoke"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"run_id": "done", "status": "completed"})
            finally:
                server.RUN_ROOT = old_root

    def test_gaussian_failure_uses_baseline_fallback_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_root = server.RUN_ROOT
            server.RUN_ROOT = Path(tmp)
            try:
                run_id = "fallback-test"
                run_dir = server.RUN_ROOT / run_id
                run_dir.mkdir()
                video = run_dir / "input.mp4"
                video.write_bytes(b"video")
                (run_dir / "job.json").write_text(json.dumps({"run_id": run_id, "input": str(video)}))
                fallback_dir = run_dir / "fallback_baseline"
                fallback_dir.mkdir()
                glb = fallback_dir / "scene.glb"
                glb.write_bytes(b"glb")
                gaussian_failure = {"status": "not measured", "error": "Gaussian unsupported", "artifacts": {}}
                baseline_success = {"status": "completed", "artifacts": {"glb": str(glb)}, "model_capabilities": {}}
                with patch.object(server, "run_benchmark", side_effect=[gaussian_failure, baseline_success]):
                    server._process_job(run_id, {"video": str(video), "reconstruction_mode": "both", "profile": "smoke"})
                manifest = json.loads((run_dir / "manifest.json").read_text())
                self.assertEqual(manifest["representation"], "mesh/glb")
                self.assertIn("fallback_baseline/scene.glb", manifest["artifacts"]["glb"])
                self.assertEqual(manifest["error"], "Gaussian unsupported")
                artifact = self.client.get(manifest["artifacts"]["glb"])
                self.assertEqual(artifact.status_code, 200)
                self.assertEqual(artifact.content, b"glb")
            finally:
                server.RUN_ROOT = old_root


if __name__ == "__main__":
    unittest.main()
