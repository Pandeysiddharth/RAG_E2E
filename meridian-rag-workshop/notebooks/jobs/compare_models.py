# Databricks notebook source
# MAGIC %md
# MAGIC # Background job: evaluate alternative chat models
# MAGIC Launched at the end of Lab 9. Evaluates the `augmented` pipeline (same prompt, same retrieval) with each chat model in
# MAGIC `chat_endpoints`, one after another to respect rate limits. Each becomes a run named `after_p9_<model>`.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(".."))
from meridian_utils import *

dbutils.widgets.text("chat_endpoints", "")
endpoints = [e.strip() for e in dbutils.widgets.get("chat_endpoints").split(",") if e.strip()]
base = load_cfg()
for ep in endpoints:
    run_name = f"after_p9_{short_model(ep).replace('-', '_')}"
    print(f"===== {ep} -> {run_name}")
    try:
        metrics = evaluate_pipeline(chat_cfg(base, ep), "augmented", run_name)
        print({k: round(v, 3) for k, v in metrics.items() if k.endswith("/mean")})
    except Exception as e:  # one model failing shouldn't stop the others
        warn(f"{ep} failed: {str(e)[:200]}")
