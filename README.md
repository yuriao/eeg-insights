# EEG Insights

Automated analysis reports on open-source EEG datasets, published as a Medium-style blog.

## Live Site

👉 **https://yuriao.github.io/eeg-insights/**

## How it works

```
Weekly cron (GitHub Actions)
  → Download EEG dataset via MOABB
  → Run ERP, ERDS, CSP+LDA analysis (MNE-Python + scikit-learn)
  → Generate figures (matplotlib)
  → Write markdown post (template + optional GPT-4o-mini discussion)
  → Commit posts/ + figures → triggers frontend rebuild → GitHub Pages
```

## Architecture

```
eeg-insights/
  pipeline/         Python analysis scripts
    analyze.py      Main pipeline — download, analyze, write post
    requirements.txt
  posts/            Auto-generated markdown reports
  frontend/         React blog site (Vite, GitHub Pages)
    public/
      posts-index.json   Auto-updated post list
      figures/           Generated plots
    src/
      pages/Home.tsx     Blog index
      pages/Post.tsx     Post reader
  .github/workflows/
    analyze.yml     Weekly cron + manual trigger
    deploy.yml      Auto-deploy on push
```

## Manual trigger

Go to **Actions → Weekly EEG Analysis → Run workflow**  
Optionally specify a dataset ID (e.g. `BNCI2014_001`) and whether to run all subjects.

## Supported datasets (via MOABB)

| Dataset | Subjects | Paradigm |
|---------|----------|---------|
| BNCI2014_001 | 9 | Motor Imagery |
| BNCI2014_004 | 9 | Motor Imagery |
| BNCI2015_001 | 12 | Motor Imagery |
| Zhou2016 | 4 | Motor Imagery |
| Schirrmeister2017 | 14 | Motor Imagery |

## Adding more datasets

Edit `pipeline/analyze.py` → `DATASETS` list. Any MOABB-compatible dataset works.

## Optional: GPT discussion

Set `OPENAI_API_KEY` in repo secrets → each post gets a 3-paragraph AI-written discussion.
Without the key, a template-based discussion is used.
