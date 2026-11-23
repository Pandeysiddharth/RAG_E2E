# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 0: Baseline and evaluation set
# MAGIC
# MAGIC **Time box:** 12 minutes
# MAGIC
# MAGIC You'll load the 40-question eval set into MLflow, look at what a naive pipeline does to Meridian's documents,
# MAGIC ask three questions and read their traces, then launch the baseline evaluation as a background job.
# MAGIC
# MAGIC **Done when:** your evaluation dataset exists and the baseline job shows as running.

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Guard cell
# MAGIC Checks that setup ran and rebuilds anything missing. If you're stuck later, set `use_checkpoint=True` and re-run.

# COMMAND ----------

from meridian_utils import *
cfg = guard_lab0(use_checkpoint=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Load the eval set into an MLflow evaluation dataset
# MAGIC Each record has `inputs` (the question) and `expectations` (the facts, source documents and rules used for scoring).
# MAGIC Storing it as a Unity Catalog evaluation dataset gives you versioning and lineage, and lets you browse it in the MLflow UI.

# COMMAND ----------

import mlflow, pandas as pd

records = load_eval_records(cfg)
print(f"{len(records)} eval records loaded")

try:
    from mlflow.genai import datasets
    ds_name = cfg.table("meridian_eval")
    try:
        ds = datasets.get_dataset(uc_table_name=ds_name)
    except Exception:
        ds = datasets.create_dataset(uc_table_name=ds_name)
    ds.merge_records(records)
    ok(f"Evaluation dataset {ds_name} ready")
except Exception as e:
    warn(f"Couldn't create the UC evaluation dataset ({e}). We'll evaluate from the JSONL records instead, which works the same.")

pd.DataFrame([{"question": r["inputs"]["question"], **r["expectations"]} for r in records]).head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC **Look closely at one record.** Notice `expected_doc_ids` uses document IDs, not chunk IDs, so recall stays comparable
# MAGIC when you change chunking later. `forbidden_content` lists strings that must never appear, like the 9-month cycle from an unapproved draft.

# COMMAND ----------

records[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Inspect the naive chunks
# MAGIC The baseline used plain text extraction and cut every document into fixed 1,000-character pieces.
# MAGIC **Find the waiver matrix. Can you still read it?**

# COMMAND ----------

chunks = spark.table(cfg.table("naive_chunks"))
display(chunks.filter("file_name LIKE '%MAT-032_v2025%'").select("chunk_id", "text"))

# COMMAND ----------

# MAGIC %md
# MAGIC Now check which corpus files produced **no chunks at all**. Anything here is invisible to the copilot.

# COMMAND ----------

raw_files = set(os.listdir(cfg.raw_path))
chunked = {r.file_name for r in chunks.select("file_name").distinct().collect()}
missing = sorted(raw_files - chunked)
for f in missing:
    fail(f"{f} produced no text")
print(f"\n{len(chunked)} of {len(raw_files)} files have chunks.")

# COMMAND ----------

# MAGIC %md
# MAGIC And a few things that **should not** be searchable but are: the unapproved draft, the travel policy and a client case file.

# COMMAND ----------

display(chunks.filter("file_name RLIKE 'DRAFT|HR-071|CASE-072'").select("file_name", "text"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Ask three questions and read the traces
# MAGIC Each call is traced. Below each answer, click the trace to see the retrieved chunks, the prompt and the token counts.
# MAGIC Try Q01, Q09 and Q23. **Which fail, and is it retrieval or generation that went wrong?**

# COMMAND ----------

rag = NaiveRAG(cfg)

questions = {
    "Q01": "How often must a high-risk CIB client's KYC be refreshed?",
    "Q09": "Who can approve a waiver of discrepancies on an LC for USD 4 million?",
    "Q23": "What additional review applies to a cash-intensive client in a UAE free zone?",
}
for qid, q in questions.items():
    print(f"--- {qid}: {q}")
    print(rag.answer(q), "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC **What you should have spotted**
# MAGIC
# MAGIC | Question | Correct answer | What usually happens | Why |
# MAGIC |---|---|---|---|
# MAGIC | Q01 | 12 months | Often right, sometimes mentions 9 months | The unapproved draft is searchable |
# MAGIC | Q09 | Country Head of Trade Operations | Often says Regional Head | The superseded 2024 matrix competes; the 2025 table is word soup |
# MAGIC | Q23 | High risk by default, 6-monthly activity review | "I don't know" or a generic answer | The scanned UAE addendum produced no text |
# MAGIC
# MAGIC Notice that in each failure the model did its job with the evidence it was given. The problems are upstream.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Launch the baseline evaluation job
# MAGIC This runs all 40 questions through the naive pipeline and scores them with five LLM judges and three code scorers.
# MAGIC It takes a few minutes, so it runs as a background job while we cover Phase 1.

# COMMAND ----------

launch_job("run_evaluation", {"pipeline": "naive", "run_name": "baseline"}, run_name="meridian-baseline-eval")

# COMMAND ----------

# MAGIC %md
# MAGIC ## While you wait (optional)
# MAGIC Run this cell again in a few minutes. When the job finishes, your baseline numbers appear here and in the MLflow experiment.
# MAGIC Open **Experiments** in the left menu, then `meridian_rag_workshop`, then the **baseline** run, and browse the per-question results.

# COMMAND ----------

print(job_status("run_evaluation"))
if table_exists(cfg, "scorecard"):
    display(show_scorecard(cfg))
