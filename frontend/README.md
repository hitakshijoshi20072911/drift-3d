# DRIFTX Showcase Frontend

Static Vite viewer adapted from the vendored AeroTwin-compatible viewer. It is branded **DRIFTX · Team ZeroError**, uses a warm yellow interface, and exposes three explicitly **PRECOMPUTED DEMO** scenes.

```bash
npm ci
npm run build
npm run dev
```

Deploy `frontend/` as a Vercel or Netlify static project. Set `VITE_DRIFTX_API_URL` to the public URL of a running DRIFTX API when upload-and-process support is required; if it is omitted, the three tracked precomputed demos still work as a static viewer. Demo artifacts are copied from repository outputs and are not live inference. New runs can be opened with the folder picker or `?data=<public-artifact-folder>`.

For a local API plus frontend, start the API on port `8123` and run `npm run dev`; Vite proxies `/api` to `http://127.0.0.1:8123`. For a production API, use an ASGI host such as Render with:

```bash
uvicorn driftx.server:app --host 0.0.0.0 --port "$PORT"
```

The API exposes `/api/health` for health checks. Configure `DRIFTX_ALLOWED_ORIGINS` on the API to the exact deployed frontend origin instead of relying on the development wildcard. GPU inference and the optional Gaussian renderer require a GPU-capable runtime such as Kaggle; static Netlify/Vercel hosting only serves the viewer.
