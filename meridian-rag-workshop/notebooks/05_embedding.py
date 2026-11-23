# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Lab 4: Embedding
# MAGIC
# MAGIC **Time box:** 8 minutes
# MAGIC
# MAGIC The Phase 3 job embedded the winning chunks with all three embedding models available on Free Edition. You'll check
# MAGIC what happens when text is longer than a model accepts, compare the models with **stricter** retrieval metrics (recall@5
# MAGIC is already perfect, so it can't tell them apart), look at speed, and choose the model for the AI Search index.
# MAGIC
# MAGIC **Done when:** you've chosen a model and the index-build job is running.

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
cfg = guard_lab4()
MODELS = [ep for ep, t in MODEL_TABLES.items() if table_exists(cfg, t)]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Will the chunks fit?
# MAGIC Every embedding model has an input limit. Text beyond it is either rejected or, worse, cut off **silently**: no error,
# MAGIC just an embedding of the first part. Here are our chunks, and our whole documents, against each model's limit.

# COMMAND ----------

chunk_tok = spark.table(cfg.table("p4_chunks")).select("text").toPandas().text.map(approx_tokens)
doc_tok = spark.table(cfg.table("silver_documents")).filter("include_in_index").select("doc_markdown").toPandas().doc_markdown.map(approx_tokens)
display(pd.DataFrame([{"model": ep.replace("databricks-", ""), "limit_tokens": MODEL_TOKEN_LIMITS[ep],
                       "chunks_over_limit": f"{int((chunk_tok > MODEL_TOKEN_LIMITS[ep]).sum())}/{len(chunk_tok)}",
                       "whole_documents_over_limit": f"{int((doc_tok > MODEL_TOKEN_LIMITS[ep]).sum())}/{len(doc_tok)}"}
                      for ep in MODELS]))
print(f"Longest chunk: about {chunk_tok.max()} tokens. Longest document: about {doc_tok.max()} tokens.")

# COMMAND ----------

# MAGIC %md
# MAGIC Our structure-aware chunks fit every model comfortably. But if you embedded **whole documents**, a third or more would
# MAGIC exceed the 512-token limit of `bge-large-en`. Let's prove the truncation is real, and silent.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Catch silent truncation
# MAGIC We embed the longest document three ways: whole, its first part only, and its last part only. If a model truncates,
# MAGIC the whole document's embedding is almost identical to the first part's, and the last part is ignored.

# COMMAND ----------

docs = spark.table(cfg.table("silver_documents")).filter("include_in_index").select("doc_id", "doc_markdown").toPandas()
longest = docs.loc[docs.doc_markdown.str.len().idxmax()]
print(f"Probe document: {longest.doc_id}, about {approx_tokens(longest.doc_markdown)} tokens")
display(pd.DataFrame([truncation_probe(cfg, ep, longest.doc_markdown) for ep in MODELS]))

# COMMAND ----------

# MAGIC %md
# MAGIC **How to read this:** for a model that reads the whole text, the full embedding is similar to **both** parts. For a
# MAGIC model that truncates, `similarity_full_vs_first_part` is close to 1.0 and the last part has noticeably less influence.
# MAGIC Anything in the tail of a too-long chunk is invisible to retrieval, and nothing warns you. If a model returned an `error`
# MAGIC instead, it rejected the long input: at least that failure is loud.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Compare models with stricter metrics
# MAGIC In Lab 3, structure-aware chunks reached recall@5 of 1.0 with `gte-large-en`. A metric at its ceiling can't separate
# MAGIC models, so we look at how **high** the right document ranks:
# MAGIC - `recall@1`, `recall@3`: is it at the very top, or in the top 3?
# MAGIC - `MRR`: average of 1 / rank of the first correct document
# MAGIC - `right_version_first`: for questions about a specific version, does that version outrank the other one?
# MAGIC
# MAGIC Each model's questions are embedded with the **same** model as its chunks.

# COMMAND ----------

ranked = {ep: retrieval_eval_ranked(cfg, MODEL_TABLES[ep], embed_cfg=endpoint_cfg(cfg, ep)) for ep in MODELS}
display(summarise_models(ranked))

# COMMAND ----------

by_cat = pd.DataFrame({ep.replace("databricks-", ""): d.groupby("category")["reciprocal_rank"].mean().round(2) for ep, d in ranked.items()})
display(by_cat.reset_index().rename(columns={"index": "category"}))

# COMMAND ----------

# MAGIC %md
# MAGIC Questions where the models disagree are the interesting ones. This lists questions where at least one model doesn't
# MAGIC rank a correct document first.

# COMMAND ----------

cmp = pd.DataFrame({ep.replace("databricks-", ""): d.set_index("question_id")["reciprocal_rank"] for ep, d in ranked.items()})
display(cmp[(cmp < 1).any(axis=1)].round(2).reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Speed
# MAGIC Every question is embedded at query time, so embedding latency adds to every answer.

# COMMAND ----------

display(pd.DataFrame([{"model": ep.replace("databricks-", ""), "median_seconds_per_query": query_latency(cfg, ep)} for ep in MODELS]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Choose the model for the index
# MAGIC Remember the noise rule: with 36 questions, a difference of one or two in a count, or a few hundredths of MRR, isn't
# MAGIC evidence. Unless another model wins clearly, staying with the one your chunks and prompts were tuned on is a sound choice.

# COMMAND ----------

INDEX_MODEL = "databricks-gte-large-en"    # change if your results show a clear winner
assert INDEX_MODEL in MODELS, f"{INDEX_MODEL} has no embedded table in this workspace"
cfg.index_embed_endpoint = INDEX_MODEL
save_cfg(cfg)
ok(f"AI Search index will use {INDEX_MODEL}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Launch the index build
# MAGIC Building the AI Search index takes several minutes, so it runs in the background while we cover Phase 5. The job
# MAGIC creates a Delta source table from the winning chunks and a Delta Sync index on top of it, embedded with your chosen model.

# COMMAND ----------

launch_job("build_index", {}, run_name="meridian-build-index")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC | You did | Why it matters |
# MAGIC |---|---|
# MAGIC | Checked chunk lengths against model limits | Text past the limit is dropped without warning |
# MAGIC | Proved truncation with a probe | Silent failures need deliberate tests |
# MAGIC | Used recall@1, MRR and a version check | A metric at its ceiling can't separate options |
# MAGIC | Measured query latency | Embedding time is paid on every question |
# MAGIC | Chose a model on evidence, not a leaderboard | Public benchmarks aren't your corpus |