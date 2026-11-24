# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 7: Retrieval
# MAGIC
# MAGIC **Time box:** 10 minutes
# MAGIC
# MAGIC Retrieval finds the right documents, but not always in the right **order**: superseded versions compete with current ones,
# MAGIC and a friendly but stale FAQ can outrank the binding policy. You'll add three improvements one at a time: a default filter
# MAGIC to current documents (lifted for historical questions), re-ranking by authority and jurisdiction, and hybrid keyword and
# MAGIC vector search. Then you'll measure them together, pick a configuration, and launch a full evaluation.
# MAGIC
# MAGIC **Done when:** you've chosen a configuration and the `after_p7` evaluation is running.

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Guard cell

# COMMAND ----------

import importlib, meridian_utils
importlib.reload(meridian_utils)        # always use the file on disk
from meridian_utils import *
self_check()                            # clear message if meridian_utils.py is out of date
import pandas as pd
cfg, index = guard_lab7()
EVAL = {json.loads(l)["tags"]["question_id"]: json.loads(l) for l in open(f"{cfg.work_path}/eval/meridian_eval_set.jsonl")}
Q = lambda qid: EVAL[qid]["inputs"]["question"]

def show(hits, cols=("doc_key", "status", "authority_rank", "jurisdiction", "score")):
    display(pd.DataFrame(hits)[[c for c in cols if c in (hits[0] if hits else {})]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Filter to current documents by default
# MAGIC Two versions of the same policy are nearly the same text, so embeddings can't reliably tell them apart: in Lab 4 the right
# MAGIC version came first in only 9 of 14 version questions. Metadata can. We filter to `status = current` by default.

# COMMAND ----------

plain = RetrievalConfig(status_default=False, authority_rerank=False)
current = RetrievalConfig(status_default=True, authority_rerank=False)
for qid in ["Q11", "Q09"]:
    plan = plan_query(cfg, Q(qid), use_llm=False)
    print(f"--- {qid}: {Q(qid)}")
    for label, rc in [("no status filter", plain), ("status default", current)]:
        hits, info = retrieve_ranked(index, plan, rc)
        print(f"   {label:17} {[(h['doc_key'], h['status']) for h in hits]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: ...but not for historical questions
# MAGIC Q15 asks what the rule was **before** 2025. It needs the superseded 2024 policy. Phase 6 flagged it as historical, so the
# MAGIC default filter is lifted.

# COMMAND ----------

plan15 = plan_query(cfg, Q("Q15"), use_llm=False)
hits, info = retrieve_ranked(index, plan15, current)
print(f"historical = {plan15.historical}; filters used: {info['filters']}")
show(hits)

# COMMAND ----------

# MAGIC %md
# MAGIC The 2024 policy (`MGB-KYC-POL-001@2024.1`, status `superseded`) should be in the results. Without the historical flag,
# MAGIC the status default would have filtered away the only document that answers Q15.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Re-rank by authority and jurisdiction
# MAGIC In Lab 5, the stale RM FAQ ("every three years") ranked **first** for a Hong Kong question, above the Hong Kong addendum.
# MAGIC A filter can't fix that: the FAQ is current. Instead we fetch 15 candidates and re-rank:
# MAGIC - similarity, normalised within the candidates
# MAGIC - **+0.05** for binding sources (policy, standard, matrix)
# MAGIC - **−0.20** for FAQs, bulletins and reference cards
# MAGIC - **+0.20** for the question's own jurisdiction addendum, because an addendum overrides the global rule

# COMMAND ----------

hk = plan_query(cfg, "How often are medium-risk clients in Hong Kong refreshed?", use_llm=False)
before, _ = retrieve_ranked(index, hk, RetrievalConfig(status_default=True, authority_rerank=False))
after, _ = retrieve_ranked(index, hk, RetrievalConfig(status_default=True, authority_rerank=True))
print("Before re-ranking:"); show(before)
print("After re-ranking:"); show(after, cols=("doc_key", "authority_rank", "jurisdiction", "similarity_norm", "adjustment", "final_score"))

# COMMAND ----------

# MAGIC %md
# MAGIC The adjustments are deliberately simple and visible: `AUTHORITY_PENALTY` and `JURISDICTION_BONUS` in `meridian_utils.py`.
# MAGIC In a bank, being able to explain **why** a source ranked first matters as much as getting it right.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Hybrid search
# MAGIC Vector search matches meaning; keyword search matches exact terms: document IDs, amounts, acronyms. Hybrid search combines
# MAGIC both. Try a question that hinges on an exact identifier.

# COMMAND ----------

exact = plan_query(cfg, "What does MGB-CR-MAT-040 say about concurrence on operational waivers?", use_llm=False)
for label, rc in [("vector only", RetrievalConfig(authority_rerank=False, hybrid=False)),
                  ("hybrid", RetrievalConfig(authority_rerank=False, hybrid=True))]:
    hits, info = retrieve_ranked(index, exact, rc)
    print(f"{label:12} ({info['mode']}): {[h['doc_key'] for h in hits]}")

# COMMAND ----------

# MAGIC %md
# MAGIC The expected source is the credit authority matrix, `MGB-CR-MAT-040`. If hybrid search isn't available in your workspace,
# MAGIC you'll see a warning and both rows use vector search; the rest of the lab still works.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Measure all of it
# MAGIC Retrieval-only evaluation over the 36 questions with known sources, adding one improvement at a time. Two extra checks:
# MAGIC `right_version_first`, and how often a FAQ, bulletin or reference card ranks first.

# COMMAND ----------

summary, grid_results = retrieval_grid(cfg, index)
display(summary)

# COMMAND ----------

by_cat = pd.DataFrame({name: d.groupby("category")["reciprocal_rank"].mean().round(2) for name, d in grid_results.items()})
display(by_cat.reset_index().rename(columns={"index": "category"}))

# COMMAND ----------

# MAGIC %md
# MAGIC **What to look for:** `right_version_first` should jump when the status default is added; `top1_is_FAQ_or_bulletin` should
# MAGIC drop when authority re-ranking is added. Hybrid search may change little on this corpus, because the context line already
# MAGIC puts doc IDs into every chunk's embedding. Apply the noise rule before calling a winner.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Choose the configuration
# MAGIC Keep an improvement only if it helps, or is neutral and makes the system easier to trust. In the dry run, hybrid search was
# MAGIC within noise of the re-rank alone on MRR, but it was the only configuration where **no** FAQ or bulletin ranked first, so it
# MAGIC stays on. If hybrid search isn't available in your workspace, the pipeline falls back to vector search automatically.

# COMMAND ----------

CHOSEN = RetrievalConfig(status_default=True, authority_rerank=True, hybrid=True, k=5, candidates=15)   # edit if your grid says otherwise
from dataclasses import asdict
with open(f"{cfg.work_path}/p7_config.json", "w") as fh:
    json.dump(asdict(CHOSEN), fh)
ok(f"Saved retrieval configuration: {asdict(CHOSEN)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Launch the after_p7 evaluation
# MAGIC Same prompt and model as before; the query plan from Phase 6 plus your retrieval configuration. It runs while we cover Phase 8.

# COMMAND ----------

launch_job("run_evaluation", {"pipeline": "retrieval", "run_name": "after_p7"}, run_name="meridian-after-p7-eval")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC | You did | Why it matters |
# MAGIC |---|---|
# MAGIC | Filtered to current documents by default | Metadata decides version; embeddings can't |
# MAGIC | Lifted the filter for historical questions | The default doesn't break questions about the past |
# MAGIC | Re-ranked by authority and jurisdiction | Binding policy beats a friendly FAQ; an addendum beats the global rule |
# MAGIC | Tried hybrid search | Exact identifiers and amounts need keyword matching |
# MAGIC | Measured each change separately | You know which change earned its place |
