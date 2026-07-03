# GitHub Pages Dashboard

This folder is a static dashboard for the DFlash Qwen 4B benchmark repo.

It is intentionally compact:

- raw benchmark JSON files stay under ignored `results/`
- long-running artifacts stay in W&B or external storage
- `docs/data/summary.json` contains only curated summaries
- the dashboard can preview pasted `bench_openai_server.py` JSON locally in the browser

## Publish

In GitHub repository settings:

1. Go to **Settings -> Pages**.
2. Set **Source** to **Deploy from a branch**.
3. Select the default branch.
4. Set the folder to `/docs`.

The site will be published at the repository GitHub Pages URL.

## Update Published Data

After a benchmark run, copy compact metrics from:

```text
results/baseline.json
results/dflash.json
```

into `docs/data/summary.json`. Do not commit large result directories or model
artifacts.
