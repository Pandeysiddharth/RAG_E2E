# Databricks notebook source
# MAGIC %md
# MAGIC # Background job: run an evaluation
# MAGIC Launched from the labs with `launch_job("run_evaluation", {...})`. You don't need to open or run this notebook yourself.
# MAGIC
# MAGIC Parameters: `pipeline` (a key of `meridian_utils.PIPELINES`) and `run_name` (the MLflow run and scorecard label).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(".."))
from meridian_utils import *

dbutils.widgets.text("pipeline", "naive")
dbutils.widgets.text("run_name", "baseline")
cfg = load_cfg()
metrics = evaluate_pipeline(cfg, dbutils.widgets.get("pipeline"), dbutils.widgets.get("run_name"))
print({k: round(v, 3) for k, v in metrics.items() if "/mean" in k})
dbutils.notebook.exit(json.dumps({k: v for k, v in metrics.items() if "/mean" in k}))
