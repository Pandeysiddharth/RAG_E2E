# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 00 Setup: run this before the workshop
# MAGIC
# MAGIC **Time:** about 10 minutes, mostly waiting. **When:** any time in the 48 hours before the session.
# MAGIC
# MAGIC This notebook prepares your Databricks Free Edition workspace for the Meridian Global Bank RAG workshop:
# MAGIC
# MAGIC 1. Creates a schema and two volumes in Unity Catalog
# MAGIC 2. Copies the corpus, eval set and checkpoints from this bundle into the volumes
# MAGIC 3. Finds the chat and embedding models available to you and tests them
# MAGIC 4. Creates your AI Search endpoint (it takes a while to come online, so we start it now)
# MAGIC 5. Builds the naive baseline chunks used in Lab 0
# MAGIC
# MAGIC Run each cell in order with **Shift + Enter**, or use **Run all** at the top. Every step prints a green tick when it succeeds.
# MAGIC If you see a red cross, read the message: it says what to do.

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create the schema and volumes
# MAGIC Free Edition gives you a catalog called `workspace`. We create a schema called `meridian_rag` inside it.

# COMMAND ----------

import os, json, shutil, importlib
import meridian_utils
importlib.reload(meridian_utils)        # always use the file on disk, never a stale cached copy
from meridian_utils import *
self_check()
cfg = resume_setup_cfg()

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.fq}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.fq}.{RAW_VOLUME} COMMENT 'Meridian corpus: raw files as received'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.fq}.{WORK_VOLUME} COMMENT 'Workshop config, eval set and checkpoints'")
ok(f"Schema {cfg.fq} with volumes '{RAW_VOLUME}' and '{WORK_VOLUME}'")
print(f"   Bundle found at: {cfg.bundle_root}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Copy the bundle into the volumes
# MAGIC The corpus goes into the `raw` volume. One document is deliberately held back in `incoming/`: you'll ingest it yourself in Lab 1.

# COMMAND ----------

import shutil
from meridian_utils import *
cfg = resume_setup_cfg()     # safe to re-run this cell on its own

def copy_tree(src, dst):
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(src)):
        if os.path.isfile(f"{src}/{f}"):
            shutil.copyfile(f"{src}/{f}", f"{dst}/{f}"); n += 1
    return n

b = cfg.bundle_root
n_corpus = copy_tree(f"{b}/corpus", cfg.raw_path)
for sub in ("eval", "checkpoints", "glossary", "incoming", "reference"):
    copy_tree(f"{b}/{sub}", f"{cfg.work_path}/{sub}")
ok(f"{n_corpus} corpus files copied to {cfg.raw_path}")
ok(f"eval set, checkpoints, glossary and held-back files copied to {cfg.work_path}")
display(dbutils.fs.ls(cfg.raw_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Find and test your models
# MAGIC Free Edition offers a subset of Foundation Model API endpoints. We pick the first available chat and embedding model.

# COMMAND ----------

from meridian_utils import *
cfg = resume_setup_cfg()     # safe to re-run this cell on its own

chat_ep, emb_ep, all_eps = detect_endpoints()
if not chat_ep or not emb_ep:
    fail("Could not find a chat and an embedding endpoint. Open Serving in the left menu and tell a helper what you see.")
    print("Available endpoints:", list(all_eps))
else:
    cfg.chat_endpoint, cfg.embed_endpoint = chat_ep, emb_ep
    answer, usage = chat([{"role": "user", "content": "Reply with the single word: ready"}], cfg, max_tokens=5)
    ok(f"Chat model {chat_ep} answered: {answer.strip()}")
    vec = embed(["Meridian Global Bank"], cfg)[0]
    ok(f"Embedding model {emb_ep} returned a vector of {len(vec)} numbers")
    save_cfg(cfg); ok(f"Config saved to {cfg.config_file}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3b. Tune embedding batch size
# MAGIC Free Edition rate-limits model calls. This finds the largest batch your workspace accepts, then leaves some headroom.
# MAGIC It takes about 15 seconds.

# COMMAND ----------

from meridian_utils import *
cfg = resume_setup_cfg()     # safe to re-run this cell on its own

diagnose_embeddings(cfg)
save_cfg(cfg); ok(f"Embedding batch size {cfg.embed_batch} saved")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create your AI Search endpoint
# MAGIC Free Edition allows one AI Search endpoint. It can take 10 to 20 minutes to come online, which is why we start it now.
# MAGIC We won't use it until Phase 5. You don't need to wait for it here.

# COMMAND ----------

from meridian_utils import *
cfg = resume_setup_cfg()     # safe to re-run this cell on its own

def ai_search_client():
    try:
        from databricks.ai_search.client import AISearchClient
        return AISearchClient(disable_notice=True)
    except ImportError:
        from databricks.vector_search.client import VectorSearchClient
        return VectorSearchClient(disable_notice=True)

vsc = ai_search_client()
existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
if AI_SEARCH_ENDPOINT in existing:
    ok(f"AI Search endpoint '{AI_SEARCH_ENDPOINT}' already exists")
elif existing:
    warn(f"You already have an AI Search endpoint ({existing[0]}). Free Edition allows one, so we'll use that.")
    cfg.ai_search_endpoint = existing[0]
else:
    vsc.create_endpoint(name=AI_SEARCH_ENDPOINT, endpoint_type="STANDARD")
    ok(f"AI Search endpoint '{AI_SEARCH_ENDPOINT}' is being created in the background")
save_cfg(cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build the naive baseline chunks
# MAGIC This is the deliberately naive starting point for Lab 0: plain text extraction and fixed 1,000-character chunks.
# MAGIC
# MAGIC If your instructor shipped pre-computed embeddings, this takes seconds and makes no model calls. Otherwise it embeds
# MAGIC the chunks and prints progress as it goes. Every embedding is cached, so **if it stops for any reason, just run the cell
# MAGIC again**: it resumes where it left off instead of starting over.

# COMMAND ----------

from meridian_utils import *
cfg = resume_setup_cfg()     # safe to re-run this cell on its own

build_naive_chunks(cfg, use_checkpoint=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Final check

# COMMAND ----------

import mlflow
from meridian_utils import *
cfg = resume_setup_cfg()     # safe to re-run this cell on its own
mlflow.set_experiment(cfg.experiment_path)
checks = {
    "Config saved": os.path.exists(cfg.config_file),
    "Corpus in raw volume": len(os.listdir(cfg.raw_path)) >= 31,
    "Eval set staged": os.path.exists(f"{cfg.work_path}/eval/meridian_eval_set.jsonl"),
    "Naive chunks table": table_exists(cfg, "naive_chunks"),
    "MLflow experiment": mlflow.get_experiment_by_name(cfg.experiment_path) is not None,
}
for k, v in checks.items():
    (ok if v else fail)(k)
print("\nAll set. See you at the workshop." if all(checks.values()) else "\nSomething is missing. Re-run the failed step or ask a helper.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Instructor only: export pre-computed embeddings
# MAGIC Run this once in **your** workspace after a successful setup. It writes the embedded baseline chunks into the bundle's
# MAGIC `checkpoints` folder. Commit that file to the workshop repository, and every participant's setup will skip embedding entirely.
# MAGIC Participants: skip this cell.

# COMMAND ----------

IS_INSTRUCTOR = False
if IS_INSTRUCTOR:
    from meridian_utils import *
    cfg = resume_setup_cfg()
    export_embedded_checkpoint(cfg, f"{cfg.bundle_root}/checkpoints")