# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Lab 5: Indexing and storage
# MAGIC
# MAGIC **Time box:** 6 minutes
# MAGIC
# MAGIC The job from Lab 4 built a managed AI Search index over the winning chunks. You'll look at what it is, query it, check
# MAGIC it returns the same results as the in-memory retrieval we've used so far, try a metadata filter, and watch a change in
# MAGIC the source table become searchable. Then you'll launch a full evaluation through the index.
# MAGIC
# MAGIC **Done when:** you've seen a filtered query and a synced change, and the `after_p5` evaluation is running.

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Guard cell
# MAGIC Checks the index build and waits briefly if the index is still syncing.

# COMMAND ----------

import importlib, meridian_utils
importlib.reload(meridian_utils)        # always use the file on disk
from meridian_utils import *
self_check()                            # clear message if meridian_utils.py is out of date
import pandas as pd
cfg, index = guard_lab5()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: What is the index?
# MAGIC A Delta Sync index is defined by a **source Delta table**, a **primary key**, the **text column** to embed, and the
# MAGIC **embedding model**. AI Search computes the embeddings itself and keeps the index in step with the table.

# COMMAND ----------

d = index.describe()
spec = d.get("delta_sync_index_spec", {})
emb = (spec.get("embedding_source_columns") or [{}])[0]
display(pd.DataFrame([
    {"property": "index", "value": d.get("name")},
    {"property": "endpoint", "value": d.get("endpoint_name")},
    {"property": "source table", "value": spec.get("source_table")},
    {"property": "primary key", "value": d.get("primary_key")},
    {"property": "embedded column", "value": emb.get("name")},
    {"property": "embedding model", "value": emb.get("embedding_model_endpoint_name")},
    {"property": "sync mode", "value": spec.get("pipeline_type")},
    {"property": "rows indexed", "value": d.get("status", {}).get("indexed_row_count")},
]))

# COMMAND ----------

# MAGIC %md
# MAGIC **Try it:** in **Catalog**, open `p5_index_source`, then the **Lineage** tab. The index appears downstream of the table,
# MAGIC which itself traces back through the chunk tables, `silver_documents`, and Bronze to the raw files. Every answer the
# MAGIC copilot gives can now be traced to a file.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Your first real query

# COMMAND ----------

q = "Who can approve a discrepancy waiver on a USD 12 million LC?"
display(pd.DataFrame(index_search(index, q, k=5))[["doc_key", "status", "score", "text"]])

# COMMAND ----------

# MAGIC %md
# MAGIC That's Q10. Look at the `status` column: the **superseded** 2024 matrix is in the results next to the current one. This is
# MAGIC the trap that made the copilot answer "Global Head of Trade" after Phase 2.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Same results as before?
# MAGIC Moving from in-memory search to a managed index shouldn't change what's retrieved: same chunks, same embedding model.
# MAGIC We run the retrieval-only evaluation both ways.

# COMMAND ----------

before = retrieval_eval_ranked(cfg, "p3_emb_structure")
through_index = index_retrieval_eval(cfg, index)
display(summarise_models({"in-memory (Lab 4)": before, "AI Search index": through_index}))

# COMMAND ----------

# MAGIC %md
# MAGIC The numbers should match closely. Small differences can come from the index's approximate nearest-neighbour search,
# MAGIC which trades a little exactness for speed at scale.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Filter on metadata
# MAGIC The chunk table carries the metadata extracted in Phase 2, so the index can filter on it **before** ranking. Here's Q10 again,
# MAGIC asking only for current documents.

# COMMAND ----------

display(pd.DataFrame(index_search(index, q, k=5, filters={"status": "current"}))[["doc_key", "status", "score", "text"]])

# COMMAND ----------

# MAGIC %md
# MAGIC The 2024 matrix is gone. One more: a Hong Kong question, restricted to Hong Kong and global documents.

# COMMAND ----------

hk = "How often are medium-risk clients refreshed?"
display(pd.DataFrame(index_search(index, hk, k=5, filters={"jurisdiction": ["HK", "GLOBAL"], "status": "current"}))[["doc_key", "jurisdiction", "score", "text"]])

# COMMAND ----------

# MAGIC %md
# MAGIC **If `MGB-FAQ-060` appears near the top**, that's the stale RM FAQ, which still says medium-risk clients are reviewed
# MAGIC "every three years". A status filter can't remove it, because the FAQ is current; it's just less authoritative than the
# MAGIC policy. Ranking by authority is a job for Phases 7 and 8.
# MAGIC
# MAGIC Filters like these are what Phase 7 builds into the pipeline: current documents by default, and the user's jurisdiction
# MAGIC when the question names one. Phase 6 will work out those filters from the question itself.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Change the source, sync, search
# MAGIC The index is a **view** of a Delta table. We add one test chunk to the table, trigger a sync, and wait until it's searchable.
# MAGIC Then we remove it again, so the evaluation isn't affected.

# COMMAND ----------

import time
SRC = cfg.table(INDEX_SOURCE)
TEST_ID = "lab5-freshness-test"
probe = "What is the trade desk mascot?"

spark.sql(f"DELETE FROM {SRC} WHERE chunk_id = '{TEST_ID}'")
spark.sql(f"""INSERT INTO {SRC} (chunk_id, file_name, doc_key, doc_id, version, status, jurisdiction, doc_type, authority_rank, chunk_index, text)
              VALUES ('{TEST_ID}', 'lab5', 'LAB5-TEST', 'LAB5-TEST', '1', 'current', 'GLOBAL', 'test', 4, 0,
                      'Lab 5 freshness test: the Meridian trade desk mascot is a grey heron called Ledger.')""")
print("Before sync:", [h["doc_key"] for h in index_search(index, probe, k=3, columns=["doc_key"])])

index.sync()
t0 = time.time()
while time.time() - t0 < 240:
    keys = [h["doc_key"] for h in index_search(index, probe, k=3, columns=["doc_key"])]
    if "LAB5-TEST" in keys:
        ok(f"Searchable after {time.time() - t0:.0f}s: {keys}"); break
    time.sleep(10)
else:
    warn("Not searchable yet after 4 minutes. Carry on; the sync will finish in the background.")

# COMMAND ----------

spark.sql(f"DELETE FROM {SRC} WHERE chunk_id = '{TEST_ID}'")
index.sync()
ok("Test chunk removed and a sync triggered. It disappears from results once the sync completes.")

# COMMAND ----------

# MAGIC %md
# MAGIC The index uses **triggered** sync: it updates when you ask it to. **Continuous** sync updates within seconds of every
# MAGIC change, at the cost of compute running all the time. Policy documents change monthly, so triggered sync, run after each
# MAGIC ingestion, is the right choice here.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Launch a full evaluation through the index
# MAGIC Same prompt, same model, retrieval through AI Search. It should score like `after_p3`, and it becomes the starting point
# MAGIC for Phases 6 to 10. It runs while we cover Phase 6.

# COMMAND ----------

launch_job("run_evaluation", {"pipeline": "indexed", "run_name": "after_p5"}, run_name="meridian-after-p5-eval")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC | You did | Why it matters |
# MAGIC |---|---|
# MAGIC | Inspected the index definition | Source table, key, model and sync mode define it |
# MAGIC | Checked parity with in-memory search | Moving to a managed index shouldn't change retrieval |
# MAGIC | Filtered on status and jurisdiction | Metadata from Phase 2 now controls what can be retrieved |
# MAGIC | Synced a change from the source table | The index is a governed view of Delta, not a separate copy to maintain |