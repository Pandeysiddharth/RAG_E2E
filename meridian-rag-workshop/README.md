# Meridian RAG Workshop bundle

Everything you need for the hands-on labs of the **Building production-grade RAG on Databricks** workshop.
All documents describe **Meridian Global Bank**, a fictional bank. Every person, company, rule and identifier is synthetic.

## Start here

1. Import this folder into your Databricks Free Edition workspace (Git folder or zip import).
2. Open `notebooks/00_setup` and click **Run all**. It takes about 10 minutes.
3. Read `docs/hands_on_guide.md` for step-by-step instructions.

## What's inside

| Folder | Contents |
|---|---|
| `corpus/` | 31 Meridian documents: PDF (one scanned), DOCX, HTML, PPTX and a photographed reference card |
| `incoming/` | One document held back on purpose. You ingest it yourself in Lab 1 |
| `eval/` | The 40-question evaluation set, as CSV (readable) and JSONL (MLflow format) |
| `glossary/` | CIB acronyms and their meanings, used in Phase 6 |
| `checkpoints/` | Pre-built outputs so nobody gets stuck if a step is slow |
| `reference/` | `corpus_manifest.csv`: ground-truth metadata for instructors and for validating your Phase 2 extraction |
| `notebooks/` | Setup, one notebook per lab, background job notebooks and the shared `meridian_utils.py` helper module |
| `docs/` | Hands-on guide and reference doc guide |

## Notebooks

| Notebook | When |
|---|---|
| `00_setup` | Before the session |
| `01_baseline` | Lab 0: baseline and evaluation set |
| `02_ingestion` | Lab 1: corpus ingestion |
| `jobs/run_evaluation` | Background job, launched from the labs |
| `jobs/parse_documents` | Background job, launched at the end of Lab 1 |

Notebooks for Labs 2 to 11 are added as the workshop build progresses.
