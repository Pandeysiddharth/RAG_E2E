# Databricks notebook source
# MAGIC %md
# MAGIC # Background job: after Lab 3
# MAGIC Launched at the end of Lab 3. It runs while the Phase 4 chapter is presented:
# MAGIC
# MAGIC 1. A full evaluation of the winning chunking strategy (pipeline `chunked`) as run **after_p3**
# MAGIC 2. Embeddings of the winning chunks with the other two embedding models, ready for the Phase 4 comparison

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(".."))
from meridian_utils import *
cfg = load_cfg()

metrics = evaluate_pipeline(cfg, "chunked", "after_p3")
print({k: round(v, 3) for k, v in metrics.items() if "/mean" in k})

# COMMAND ----------

source = spark.table(cfg.table(p3_winner_table(cfg))).drop("embedding")
source.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("p4_chunks"))
_, _, eps = detect_endpoints()
others = [e for e in EMBED_PREFS if e in eps and e != cfg.embed_endpoint]
for ep in others:
    short = ep.replace("databricks-", "").split("-")[0]          # gte, bge, qwen3
    try:
        embed_chunk_table(endpoint_cfg(cfg, ep), "p4_chunks", f"p4_emb_{short}")
    except Exception as e:  # one model failing shouldn't stop the other
        warn(f"Embedding with {ep} failed: {str(e)[:200]}")
