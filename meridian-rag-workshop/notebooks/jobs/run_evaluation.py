# Databricks notebook source
# MAGIC %md
# MAGIC # Background job: run an evaluation
# MAGIC Launched from the labs with `launch_job("run_evaluation", {...})`. You don't need to open or run this notebook yourself.
# MAGIC
# MAGIC Parameters: `pipeline` (which pipeline from `meridian_utils.PIPELINES`) and `run_name` (the MLflow run and scorecard label).

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(".."))
from meridian_utils import *
import mlflow

dbutils.widgets.text("pipeline", "naive")
dbutils.widgets.text("run_name", "baseline")
pipeline, run_name = dbutils.widgets.get("pipeline"), dbutils.widgets.get("run_name")

cfg = load_cfg()
mlflow.set_experiment(cfg.experiment_path)
rag = PIPELINES[pipeline](cfg)
records = load_eval_records(cfg)
print(f"Evaluating pipeline '{pipeline}' on {len(records)} questions as run '{run_name}'")

# COMMAND ----------

with mlflow.start_run(run_name=run_name):
    mlflow.log_params({"pipeline": pipeline, "chat_endpoint": cfg.chat_endpoint, "embed_endpoint": cfg.embed_endpoint})
    results = mlflow.genai.evaluate(data=records, predict_fn=rag.answer, scorers=get_scorers(cfg))

print(results.metrics)
record_scorecard(cfg, run_name, results.metrics)
dbutils.notebook.exit(json.dumps({k: v for k, v in results.metrics.items() if isinstance(v, (int, float))}))
