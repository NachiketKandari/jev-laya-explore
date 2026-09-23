# Laya Demo Web App

Minimal Next.js viewer for Laya customer-query classification results.

## Run

```bash
npm install
npm run dev
```

Then open http://localhost:3000.

## Data

The app fetches `public/results.json` client-side at runtime, so
re-running the classifier updates the app with no rebuild.

Regenerate the data with either:

```bash
uv run scripts/classify_demo_data.py
```

from the repo root (writes into `web/public/`), or via the notebook
(see repo root).
