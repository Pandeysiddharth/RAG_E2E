# Databricks notebook source
# MAGIC %md
# MAGIC # Background job: build the AI Search index
# MAGIC Launched at the end of Lab 4. It runs while the Phase 5 chapter is presented:
# MAGIC
# MAGIC 1. Writes the winning chunks to a Delta source table with Change Data Feed enabled
# MAGIC 2. Waits for your AI Search endpoint to be online
# MAGIC 3. (Re)creates a Delta Sync index over that table, embedded by the model chosen in Lab 4, and waits until it's ready

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os, sys, time
sys.path.append(os.path.abspath(".."))
from meridian_utils import *
cfg = load_cfg()
model = cfg.index_embed_endpoint or cfg.embed_endpoint
source, index_name, endpoint = cfg.table(INDEX_SOURCE), cfg.table(INDEX_NAME), cfg.ai_search_endpoint
print(f"Source {source} -> index {index_name} on endpoint {endpoint}, embedded with {model}")

spark.table(cfg.table("p4_chunks")).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(source)
spark.sql(f"ALTER TABLE {source} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
ok(f"{source}: {spark.table(source).count()} chunks, Change Data Feed enabled")

# COMMAND ----------

vsc = ai_search_client()
deadline = time.time() + 25 * 60
while True:
    state = vsc.get_endpoint(endpoint).get("endpoint_status", {}).get("state")
    if state == "ONLINE":
        ok(f"Endpoint {endpoint} is online"); break
    if time.time() > deadline:
        raise RuntimeError(f"Endpoint {endpoint} still {state} after 25 minutes")
    print(f"  endpoint {state}, waiting ..."); time.sleep(30)

existing = [i.get("name") for i in vsc.list_indexes(endpoint).get("vector_indexes", [])]
if index_name in existing:
    vsc.delete_index(endpoint_name=endpoint, index_name=index_name)
    print("  deleted the previous index; waiting for it to disappear ...")
    while index_name in [i.get("name") for i in vsc.list_indexes(endpoint).get("vector_indexes", [])]:
        time.sleep(10)

index = vsc.create_delta_sync_index(endpoint_name=endpoint, index_name=index_name, source_table_name=source,
                                    pipeline_type="TRIGGERED", primary_key="chunk_id",
                                    embedding_source_column="text", embedding_model_endpoint_name=model)
ok("Index created; first sync running")

# COMMAND ----------

deadline = time.time() + 25 * 60
while True:
    status = index.describe().get("status", {})
    if status.get("ready"):
        ok(f"Index ready: {status.get('indexed_row_count', '?')} rows indexed"); break
    if time.time() > deadline:
        raise RuntimeError(f"Index not ready after 25 minutes: {status}")
    print(f"  {status.get('detailed_state', 'syncing')} ..."); time.sleep(30)

with open(f"{cfg.work_path}/p5_index.json", "w") as fh:
    json.dump({"index_name": index_name, "endpoint": endpoint, "source": source, "embedding_model": model}, fh)
