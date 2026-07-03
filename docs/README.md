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

The dashboard currently publishes compact H100 comparison summaries from:

```text
results/comparison_h100_c32.txt
results/compare_qwen35_4b_h100_c32_b8.txt
```

After future benchmark runs, copy compact metrics from `results/*.json` or
`results/*.txt` into `docs/data/summary.json`. Do not commit large result
directories or model artifacts.
