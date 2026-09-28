# DRIFTX local testing and free-tier deployment runbook

This guide covers the repository at `hitakshijoshi20072911/drift-3d` and the current production behavior.

## Important reality about “free” GPU hosting

A fully free, always-on public GPU inference API is not available from the services in this guide.

- **Netlify Free** is appropriate for the static Vite viewer and its precomputed demos.
- **Render Free** can host a lightweight FastAPI demo/status service, but it has a 512 MB free instance, sleeps after 15 minutes without traffic, and has an ephemeral filesystem. It is not suitable for loading the DA3 model or storing uploaded reconstruction artifacts persistently.
- **Google Colab Free** and **Kaggle free GPU notebooks** are suitable for temporary, manual GPU runs. They are not reliable public API hosts: runtimes are temporary, resources are not guaranteed, and sessions end.
- A real always-on upload → GPU inference → artifact API requires a paid GPU VM/service or a machine you own.

Therefore, the **no-cost stable deployment** is:

> Netlify static viewer + tracked precomputed demos, with local/Kaggle/Colab used for occasional GPU inference.

Do not put private drone footage on a public free notebook or public demo URL.

---

## 1. Prerequisites

Install:

- Git
- Python 3.9–3.13; Python 3.11 or 3.12 is recommended
- Node.js 18.14+ and npm
- An NVIDIA GPU is optional for viewer/API tests, but required for practical live DA3 inference

Clone the repository:

```bash
git clone https://github.com/hitakshijoshi20072911/drift-3d.git
cd drift-3d
```

Windows PowerShell uses the same commands unless noted below.

---

## 2. Local installation

Create and activate a virtual environment:

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the project and test tools:

```bash
python -m pip install -e ".[test]"
```

This project has large ML dependencies. If you have an NVIDIA GPU, install a matching PyTorch build first using the selector at [pytorch.org](https://pytorch.org/get-started/locally/), then install the project dependencies.

Run the runtime diagnostic:

```bash
python -m driftx doctor --json
```

A CPU-only machine may report warnings. That does not prevent the static viewer or API contract tests from running.

---

## 3. Run all local automated checks

Run the DRIFTX package tests:

```bash
python -m pytest -q tests
```

Run the same suite through the standard-library test runner:

```bash
python -m unittest discover -s tests -v
```

Compile-check the Python source:

```bash
python -m compileall -q driftx src app_live.py da3_streaming
```

Build the Python package:

```bash
python -m pip install build hatchling hatch-vcs
python -m build --sdist --wheel --no-isolation
```

Check the frontend production build:

```bash
cd frontend
npm ci
npm run build
cd ..
```

The Vite build copies `frontend/demo/` into `frontend/dist/demo/`; verify that the static demo fallback is present:

```bash
test -f frontend/dist/demo/index.json
# Windows PowerShell:
# Test-Path frontend/dist/demo/index.json
```

Do not commit `frontend/node_modules/` or generated `frontend/dist/` files unless the repository explicitly asks for them.

---

## 4. Test the local API and viewer

Open two terminals. Activate `.venv` in both.

### Terminal A: API

From the repository root:

```bash
python -m driftx server --host 127.0.0.1 --port 8123
```

Check the health endpoint:

```bash
curl -fsS http://127.0.0.1:8123/api/health
curl -fsS http://127.0.0.1:8123/api/demos
```

Check each real GLB artifact:

```bash
curl -fI http://127.0.0.1:8123/api/demos/test3/artifacts/scene.glb
curl -fI http://127.0.0.1:8123/api/demos/test6/artifacts/scene.glb
curl -fI http://127.0.0.1:8123/api/demos/test7/artifacts/scene.glb
```

Windows PowerShell equivalents:

```powershell
Invoke-RestMethod http://127.0.0.1:8123/api/health
Invoke-RestMethod http://127.0.0.1:8123/api/demos
curl.exe -I http://127.0.0.1:8123/api/demos/test3/artifacts/scene.glb
```

### Terminal B: Vite viewer

```bash
cd frontend
npm ci
npm run dev
```

Open the printed URL, normally `http://localhost:5173`.

Verify:

1. The page loads without a JavaScript error.
2. The API status changes to connected.
3. Each of `test3`, `test6`, and `test7` opens a 3D scene.
4. The quality/evidence panel loads its preview/depth evidence.
5. The viewer tools work: height, points, distance, area, profile, sight line, screenshot, and report export.
6. Stop Terminal A and refresh. The page should still show the three static demos because the built frontend contains `demo/`.

Vite proxies `/api` to `http://127.0.0.1:8123` during local development.

---

## 5. Run a real local reconstruction

Only do this after the API/viewer smoke test passes.

Use a local GPU if possible. The first run downloads the selected checkpoint and may take time.

```bash
python -m driftx benchmark \
  --video input/test.mp4 \
  --output outputs/benchmarks/local_smoke \
  --model depth-anything/DA3-LARGE-1.1 \
  --device auto \
  --profile smoke \
  --reconstruction-mode baseline
```

Windows PowerShell:

```powershell
python -m driftx benchmark `
  --video input/test.mp4 `
  --output outputs/benchmarks/local_smoke `
  --model depth-anything/DA3-LARGE-1.1 `
  --device auto `
  --profile smoke `
  --reconstruction-mode baseline
```

Confirm these outputs exist:

```bash
test -f outputs/benchmarks/local_smoke/scene.glb
test -f outputs/benchmarks/local_smoke/run_report.json
test -f outputs/benchmarks/local_smoke/results.npz
```

For a first run on limited VRAM, keep `--profile smoke` and `--reconstruction-mode baseline`. Gaussian mode requires a Gaussian-capable checkpoint and the optional `gsplat` renderer; it is not required for the baseline GLB path.

---

## 6. Deploy the static viewer to Netlify Free

This is the recommended free deployment.

### 6.1 Deploy from the Netlify dashboard

1. Push your latest code to GitHub.
2. Sign in at [Netlify](https://app.netlify.com/).
3. Choose **Add new site → Import an existing project**.
4. Select GitHub and choose `hitakshijoshi20072911/drift-3d`.
5. Set:
   - **Base directory:** `frontend`
   - **Build command:** `npm ci && npm run build`
   - **Publish directory:** `frontend/dist` if Netlify interprets it from repository root, or `dist` if it interprets it relative to the base directory. Use the previewed build output to confirm.
6. Deploy the site.
7. Open the generated `https://<site>.netlify.app` URL.

The checked-in `frontend/netlify.toml` already defines:

```toml
[build]
  command = "npm ci && npm run build"
  publish = "dist"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

With **Base directory = `frontend`**, `publish = "dist"` is the correct value.

### 6.2 Netlify environment variables

For the free static demo deployment, leave `VITE_DRIFTX_API_URL` empty. The viewer then uses its static precomputed demos and does not depend on a sleeping API.

If you later have a reachable API, add this variable in **Site configuration → Environment variables**:

```text
VITE_DRIFTX_API_URL=https://your-api.example.com
```

After changing it, trigger a new deploy because Vite embeds `VITE_*` variables into the frontend at build time.

Never put secrets in `VITE_*` variables; they are visible in the browser bundle.

### 6.3 Netlify validation checklist

- Open the site in a private/incognito window.
- Hard-refresh once.
- Open all three demo cards.
- In browser DevTools → Network, verify `demo/index.json` and the three `model.glb` files return 200.
- Directly open a deep URL if the app uses one; the SPA rewrite should return `index.html` rather than a 404.
- Check the Netlify deploy log has a successful `npm ci` and `npm run build`.

Netlify’s free plan and limits can change. Check the [official pricing page](https://www.netlify.com/pricing/) before publishing a public demo.

---

## 7. Deploy a limited demo API to Render Free

Render Free is useful for checking that the API is reachable and for serving the repository’s precomputed demo artifacts. It is **not** a free GPU backend and should not be used for persistent uploaded runs.

### 7.1 Create the service

1. Sign in at [Render](https://dashboard.render.com/).
2. Choose **New → Web Service**.
3. Connect `hitakshijoshi20072911/drift-3d`.
4. Set the root directory to the repository root.
5. Set:
   - **Runtime:** Python 3
   - **Build command:** `pip install -r requirements-render.txt`
   - **Start command:** `uvicorn driftx.server:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free
6. Add environment variables:

```text
DRIFTX_ALLOWED_ORIGINS=https://<your-site>.netlify.app
DRIFTX_RUN_ROOT=/tmp/driftx-runs
DRIFTX_MAX_UPLOAD_BYTES=524288000
```

7. Add the health check path `/api/health` if the Render UI asks for one.
8. Deploy and wait for the first build to finish.

Test it:

```bash
curl -fsS https://<your-service>.onrender.com/api/health
curl -fsS https://<your-service>.onrender.com/api/demos
```

Render Free may take about a minute to wake after idle. Its local filesystem is ephemeral, so uploaded videos and generated artifacts are lost after restart, redeploy, or sleep. Do not treat it as durable storage.

### 7.2 What not to do on Render Free

Do not expect this configuration to run `driftx benchmark` successfully. The DA3 model and its ML dependencies are too large for the free CPU instance, and Render Free has no GPU. If you set `VITE_DRIFTX_API_URL` to this Render service, the viewer’s upload button may reach the API but live inference is not a supported free-tier deployment.

Use Render only for API/demo validation, or leave `VITE_DRIFTX_API_URL` empty for the stable static Netlify deployment.

---

## 8. Use a free temporary GPU with Kaggle

Kaggle is the better free GPU notebook option for a manual run. It is not an always-on API host.

1. Create/sign in to a [Kaggle](https://www.kaggle.com/) account.
2. Create a new Notebook.
3. Open **Notebook settings** and choose **Accelerator → GPU**.
4. Add this repository as a GitHub source or clone it in a notebook cell:

```python
!git clone https://github.com/hitakshijoshi20072911/drift-3d.git
%cd drift-3d
```

5. Install the dependencies. Start with the repository’s Kaggle notebook if available (`cloud/kaggle_driftx_gaussian.ipynb`) and adapt its install cells to the current commit. For the baseline path, install the matching PyTorch build first, then:

```python
!pip install -e ".[test]"
```

6. Confirm the GPU:

```python
!nvidia-smi
!python -m driftx doctor --json
```

7. Upload a short test video to the notebook or copy it from a private dataset. Do not commit private video to the public GitHub repository.
8. Run the smallest baseline test:

```python
!python -m driftx benchmark \
  --video /kaggle/working/test.mp4 \
  --output /kaggle/working/driftx_run \
  --model depth-anything/DA3-LARGE-1.1 \
  --device cuda \
  --profile smoke \
  --reconstruction-mode baseline
```

9. Check the output:

```python
!test -f /kaggle/working/driftx_run/scene.glb
!cat /kaggle/working/driftx_run/run_report.json
```

10. Download `scene.glb`, `run_report.json`, `results.npz`, and evidence files from the Kaggle output panel.
11. Inspect the downloaded GLB locally in the DRIFTX viewer using the folder picker, or publish only intentionally public artifacts to a static host.

Kaggle documents free GPU availability but also queues and temporary sessions. Keep checkpoints and outputs in a private dataset or download them before the session ends.

---

## 9. Use Google Colab Free instead

Colab Free is also suitable for occasional manual GPU runs:

1. Open Colab and create a notebook.
2. Choose **Runtime → Change runtime type → T4 GPU** when available.
3. Clone the repository and install dependencies:

```python
!git clone https://github.com/hitakshijoshi20072911/drift-3d.git
%cd drift-3d
!pip install -e ".[test]"
!nvidia-smi
```

4. Upload a short video through the Files panel or mount Drive only if the data is private and you understand the notebook permissions.
5. Run the same `driftx benchmark` command as in the Kaggle section, using the actual path shown by Colab.
6. Download the output artifacts before disconnecting.

Do not expose the Colab runtime as a public production web server. Colab’s official policy restricts free runtimes from being used primarily as a web UI or file-hosting service, and its GPU availability/runtime limits are dynamic.

---

## 10. Recommended deployment choices

### Stable and completely free demo

```text
Netlify Free
  └── Vite viewer + copied static precomputed demos
```

Leave `VITE_DRIFTX_API_URL` unset.

### Free demo plus temporary inference

```text
Netlify Free
  └── static viewer
Kaggle or Colab Free
  └── manually run one reconstruction, download artifacts, inspect/publish intentionally
```

### Public live upload-and-process product

```text
Netlify frontend
  └── paid or self-hosted authenticated API
      └── persistent storage
      └── GPU worker/service
```

This last architecture cannot honestly be promised as permanently free. Do not use Render Free as the GPU worker.

---

## Official references

- [Netlify: Vite on Netlify](https://docs.netlify.com/build/frameworks/framework-setup-guides/vite/)
- [Netlify pricing](https://www.netlify.com/pricing/)
- [Render: Deploy a FastAPI app](https://render.com/docs/deploy-fastapi)
- [Render: Deploy for Free limitations](https://render.com/docs/free)
- [Google Colab FAQ and resource limits](https://research.google.com/colaboratory/faq.html)
- [Kaggle Notebooks and free GPU documentation](https://www.kaggle.com/docs/notebooks)
