# Databricks notebook source
# MAGIC %md
# MAGIC # Background job: after Lab 2
# MAGIC Launched at the end of Lab 2. It runs while the Phase 3 chapter is presented, and does three things:
# MAGIC
# MAGIC 1. Builds three chunking variants from `silver_documents` for Phase 3: fixed, recursive and structure-aware
# MAGIC 2. Embeds the fixed-size chunks (same chunking as the baseline, so only Phase 2's changes differ)
# MAGIC 3. Evaluates that pipeline as run **after_p2**, so the scorecard shows what parsing, cleaning and filtering were worth

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(".."))
from meridian_utils import *
cfg = load_cfg()

build_chunk_tables(cfg)
embed_chunk_table(cfg, "p2_chunks_fixed", "p2_chunks_embedded")

# COMMAND ----------

metrics = evaluate_pipeline(cfg, "parsed", "after_p2")
print({k: round(v, 3) for k, v in metrics.items() if "/mean" in k})
