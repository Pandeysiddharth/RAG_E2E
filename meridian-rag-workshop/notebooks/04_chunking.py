# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Lab 3: Chunking
# MAGIC
# MAGIC **Time box:** 12 minutes
# MAGIC
# MAGIC The Phase 2 follow-up job cut the 28 clean documents three ways: fixed-size, recursive and structure-aware. You'll look at
# MAGIC where each strategy puts its boundaries, check which ones break tables, embed the variants, and compare them with a
# MAGIC **retrieval-only evaluation**: no LLM calls, just "did the right document reach the top 5?". Then you'll launch a full
# MAGIC evaluation of the winner.
# MAGIC
# MAGIC **Done when:** you've picked a winning strategy and the follow-up job is running.

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Guard cell
# MAGIC Checks the Phase 2 follow-up job and rebuilds anything missing.

# COMMAND ----------

import importlib, meridian_utils
importlib.reload(meridian_utils)        # always use the file on disk
from meridian_utils import *
self_check()                            # clear message if meridian_utils.py is out of date
import pandas as pd
cfg = guard_lab3()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Where do the boundaries fall?
# MAGIC Q02 asks what defines an ultimate beneficial owner. The answer is two facts in one paragraph: **25% or more**, *or*
# MAGIC **otherwise exercises control**. After Phase 2, the copilot gave the first and missed the second. Let's see why.

# COMMAND ----------

def chunks_for(table, file_prefix, contains=None):
    df = spark.table(cfg.table(table)).toPandas()
    df = df[df.file_name.str.startswith(file_prefix)].sort_values("chunk_index")
    if contains:
        df = df[df.text.str.contains(contains, case=False, regex=False)]
    return df[["chunk_index", "text"]]

for name, table in [("fixed", "p2_chunks_fixed"), ("recursive", "p2_chunks_recursive"), ("structure", "p2_chunks_structure")]:
    hits = chunks_for(table, "MGB-KYC-STD-011", "25%")
    print(f"===== {name}: chunk(s) containing '25%' =====")
    for _, r in hits.iterrows():
        has_control = "otherwise exercises control" in r.text
        print(f"[chunk {r.chunk_index}]  contains 'otherwise exercises control': {has_control}")
        print(r.text[:700], "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC **What to notice:**
# MAGIC - **Fixed-size** chunks cut at a character count, wherever that falls. Check whether your fixed chunk keeps both UBO facts
# MAGIC   together, and notice that it also runs on into the next section about trusts: one chunk, two topics.
# MAGIC - **Recursive** chunks prefer paragraph boundaries, and overlap so a fact near a boundary appears in both neighbours.
# MAGIC - **Structure-aware** chunks hold exactly one section ("Ownership and control threshold") and start with a context line:
# MAGIC   document, version, status, jurisdiction and effective date.
# MAGIC
# MAGIC If the control clause is in the same chunk for every strategy, Q02's miss after Phase 2 came from elsewhere, possibly
# MAGIC run-to-run noise. That's worth checking before blaming chunking.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Sizes, and do tables survive?
# MAGIC A **table fragment** is a chunk containing table rows but not the table's header, so the rows have lost their column meanings.

# COMMAND ----------

report = pd.DataFrame([{"strategy": n, **chunk_report(cfg, t)} for n, t in
                       [("fixed", "p2_chunks_fixed"), ("recursive", "p2_chunks_recursive"), ("structure", "p2_chunks_structure")]])
display(report)

# COMMAND ----------

print(chunks_for("p2_chunks_structure", "MGB-TF-MAT-032_v2025", "Primary approver").text.iloc[0][:900])

# COMMAND ----------

# MAGIC %md
# MAGIC Structure-aware chunks never split a table: the waiver matrix arrives whole, with its header and the context line telling
# MAGIC the model it's the **current, 2025.1** version. The trade-off is more, smaller chunks, which you'll see matters in Step 5.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Embed the two new variants
# MAGIC The fixed-size chunks were embedded by the Phase 2 job. Embedding the other two takes a minute or two; the cache means
# MAGIC nothing is embedded twice.

# COMMAND ----------

embed_chunk_table(cfg, "p2_chunks_recursive", "p3_emb_recursive")
embed_chunk_table(cfg, "p2_chunks_structure", "p3_emb_structure")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Compare strategies with retrieval-only evaluation
# MAGIC For each of the 36 questions that name their source documents, we embed the question, take the top 5 chunks, and check
# MAGIC whether the expected documents are there. It takes seconds and costs no LLM calls, so it's ideal for comparing options
# MAGIC before paying for a full evaluation. The baseline's naive chunks are included for reference.

# COMMAND ----------

results = {name: retrieval_eval(cfg, table) for name, table in CHUNK_TABLES.items()}
overall, by_cat = summarise_retrieval(results)
display(overall)

# COMMAND ----------

display(by_cat)

# COMMAND ----------

# MAGIC %md
# MAGIC **How to read this:**
# MAGIC - `recall@5`: share of expected documents found in the top 5, averaged over questions.
# MAGIC - `MRR`: how high the first correct document ranks (1.0 means always first).
# MAGIC - `avg_distinct_docs_in_top5`: smaller chunks can mean several chunks from the **same** document fill the top 5, crowding
# MAGIC   out the second document a multi-hop question needs.
# MAGIC
# MAGIC Remember the noise rule from Phase 2: with five or six questions per category, a difference of one question is noise.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Read a failure
# MAGIC Pick a question where the strategies disagree, and look at what each one retrieved.

# COMMAND ----------

QID = "Q31"   # try Q02, Q09, Q31 or Q33
for name, df in results.items():
    row = df[df.question_id == QID].iloc[0]
    print(f"{name:18} recall {row.recall:.2f}   top 5: {row.top_docs}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Pick the winner
# MAGIC Choose the strategy you'd ship. Recall matters most; break ties on MRR. The choice is saved, and the `chunked` pipeline
# MAGIC uses it from now on.

# COMMAND ----------

WINNER = "structure"     # change if your results say otherwise: "fixed", "recursive" or "structure"
with open(f"{cfg.work_path}/p3_winner.json", "w") as fh:
    json.dump({"strategy": WINNER, "table": CHUNK_TABLES[WINNER]}, fh)
ok(f"Winning strategy: {WINNER} ({CHUNK_TABLES[WINNER]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Launch the follow-up job
# MAGIC It runs a full evaluation of the winning strategy as run `after_p3`, then embeds the winning chunks with the other two
# MAGIC embedding models for Phase 4. It runs while we cover Phase 4.

# COMMAND ----------

launch_job("phase3_followup", {}, run_name="meridian-phase3-followup")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC | You did | Why it matters |
# MAGIC |---|---|
# MAGIC | Looked at where boundaries fall | A fixed cut can split one fact across two chunks |
# MAGIC | Checked for table fragments | Rows without headers lose their meaning |
# MAGIC | Added a context line to every chunk | Each chunk says which document, version and jurisdiction it's from |
# MAGIC | Compared strategies without an LLM | Retrieval-only evaluation is fast and cheap for choosing between options |
# MAGIC | Watched distinct documents in the top 5 | Smaller chunks can crowd out the second source a multi-hop answer needs |

# COMMAND ----------

print([t for t in ["p4_emb_bge", "p4_emb_qwen3"] if table_exists(cfg, t)])