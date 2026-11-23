# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Lab 1: Corpus ingestion
# MAGIC
# MAGIC **Time box:** 10 minutes
# MAGIC
# MAGIC You'll land Meridian's raw files in a Bronze Delta table with Auto Loader, check whether the corpus actually covers
# MAGIC the questions users ask, ingest a missing document incrementally, and decide what's in scope. Then you'll launch
# MAGIC document parsing as a background job for Phase 2.
# MAGIC
# MAGIC **Done when:** `bronze_documents` has 32 rows, `source_registry` exists, and the parse job shows as running.

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Guard cell and status check
# MAGIC Shows whether your baseline evaluation from Lab 0 has finished. It doesn't block this lab either way.

# COMMAND ----------

from meridian_utils import *
from pyspark.sql import functions as F
cfg = guard_lab1()
BRONZE = cfg.table("bronze_documents")
CKPT = f"{cfg.work_path}/_checkpoints/bronze_documents"

# Set to True only if you want to start this lab again from scratch
RESET = False
if RESET:
    spark.sql(f"DROP TABLE IF EXISTS {BRONZE}")
    dbutils.fs.rm(CKPT, recurse=True)
    for f in os.listdir(cfg.raw_path):
        if f.startswith("MGB-KYC-ADD-UK"):
            os.remove(f"{cfg.raw_path}/{f}")
    ok("Lab 1 reset")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: See what landed
# MAGIC Before building anything, look at the raw landing zone. `read_files` with the `binaryFile` format reads every file as bytes plus metadata.

# COMMAND ----------

display(spark.sql(f"""
  SELECT lower(regexp_extract(path, '\\\\.([^.]+)$', 1)) AS file_type,
         count(*)                                        AS files,
         round(sum(length) / 1024)                       AS total_kb
  FROM read_files('{cfg.raw_path}', format => 'binaryFile')
  GROUP BY 1 ORDER BY files DESC"""))

# COMMAND ----------

# MAGIC %md
# MAGIC Five file types, and one of them is an image. Note also the file names: most follow the `doc_id_title` convention, but not all of them do.

# COMMAND ----------

display(spark.sql(f"SELECT path, length, modificationTime FROM read_files('{cfg.raw_path}', format => 'binaryFile') ORDER BY path"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Ingest into Bronze with Auto Loader
# MAGIC Auto Loader (`cloudFiles`) tracks which files it has already processed in its checkpoint. With `trigger(availableNow=True)`
# MAGIC it processes everything new and then stops, so the same code works for the first load and for every later top-up.
# MAGIC
# MAGIC We keep the raw bytes in Bronze. That means Phase 2 can re-parse with a better parser without re-ingesting anything,
# MAGIC and every answer can be traced back to the exact file version it came from.

# COMMAND ----------

def ingest_bronze():
    """Incremental load: append only files Bronze hasn't seen (same path and same content hash).
    Re-running is safe; changed files (new hash) are picked up as new versions.
    In production you'd use Auto Loader (cloudFiles) for this; see the deck."""
    src = (spark.read.format("binaryFile").load(cfg.raw_path)
           .select("path", "modificationTime", "length", "content")
           .withColumn("file_name", F.element_at(F.split("path", "/"), -1))
           .withColumn("file_ext", F.lower(F.element_at(F.split("file_name", r"\."), -1)))
           .withColumn("doc_id_hint", F.when(F.col("file_name").startswith("MGB-"), F.split("file_name", "_")[0]))
           .withColumn("content_sha256", F.sha2("content", 256))
           .withColumn("ingested_at", F.current_timestamp()))
    if spark.catalog.tableExists(BRONZE):
        src = src.join(spark.table(BRONZE).select("path", "content_sha256"), ["path", "content_sha256"], "left_anti")
    n = src.count()
    if n:
        src.write.mode("append").saveAsTable(BRONZE)
    ok(f"Processed {n} new file(s). {BRONZE} now has {spark.table(BRONZE).count()} rows.")
    return n

ingest_bronze()

# COMMAND ----------

display(spark.table(BRONZE).select("file_name", "file_ext", "doc_id_hint", "length", "content_sha256", "ingested_at"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Does the corpus cover the questions?
# MAGIC Corpus composition starts from the questions users will ask, not from the documents you happen to have.
# MAGIC Our eval set says which document each question needs. Let's check every one of them is actually in Bronze.

# COMMAND ----------

records = load_eval_records(cfg)
needed = {}
for i, r in enumerate(records, start=1):
    for d in r["expectations"]["expected_doc_ids"]:
        needed.setdefault(base_doc_id(d), []).append(f"Q{i:02d}")

have = {r.doc_id_hint for r in spark.table(BRONZE).select("doc_id_hint").distinct().collect() if r.doc_id_hint}
gaps = {d: qs for d, qs in needed.items() if d not in have}
print(f"The eval set needs {len(needed)} distinct documents; Bronze has {len(have & set(needed))} of them.\n")
for d, qs in gaps.items():
    fail(f"{d} is missing. Questions that can never be answered: {', '.join(qs)}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Your RAG can't retrieve what you never ingested.** No amount of chunking or prompt tuning fixes Q25 while the UK addendum
# MAGIC isn't in the corpus. The file has just arrived in the `incoming` folder. Let's land it in the raw volume and run the
# MAGIC **same** ingestion code again.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Incremental ingestion
# MAGIC Watch the count: Auto Loader should process exactly **one** new file, not all 32.

# COMMAND ----------

import shutil
for f in os.listdir(f"{cfg.work_path}/incoming"):
    shutil.copyfile(f"{cfg.work_path}/incoming/{f}", f"{cfg.raw_path}/{f}")
    ok(f"Landed {f} in the raw volume")

ingest_bronze()

# COMMAND ----------

have = {r.doc_id_hint for r in spark.table(BRONZE).select("doc_id_hint").distinct().collect() if r.doc_id_hint}
still_missing = [d for d in needed if d not in have]
(ok("Every document the eval set needs is now in Bronze") if not still_missing else fail(f"Still missing: {still_missing}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Decide what's in scope
# MAGIC The copilot is for onboarding and trade operations. Some files in the landing zone don't belong in its knowledge at all.
# MAGIC We record the decision in a `source_registry` table rather than deleting files, so the choice is visible and reversible.
# MAGIC
# MAGIC In Phase 2 we'll go further and filter by content: drafts, personal data and near-duplicates.

# COMMAND ----------

IN_SCOPE_DOMAINS = {"KYC", "SAN", "TF", "CR", "OPS", "FAQ", "GLO", "COM", "DRAFT", "CASE"}  # Phase 2 decides on DRAFT and CASE

registry = (spark.table(BRONZE)
    .select("file_name", "doc_id_hint", "file_ext", "content_sha256")
    .withColumn("domain_code", F.split(F.col("doc_id_hint"), "-")[1])
    .withColumn("in_scope", F.coalesce(F.col("domain_code").isin(list(IN_SCOPE_DOMAINS)), F.lit(True)))
    .withColumn("scope_reason", F.when(F.col("doc_id_hint").isNull(), "No doc ID in file name: keep, check in Phase 2")
                                 .when(F.col("in_scope"), "In copilot domain")
                                 .otherwise("Outside copilot domain")))
registry.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("source_registry"))
display(spark.table(cfg.table("source_registry")).orderBy("in_scope", "file_name"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Try it:** open `bronze_documents` in **Catalog** (left menu), then the **Lineage** tab. Unity Catalog records that
# MAGIC `source_registry` was derived from it, and the volume it was loaded from. Every downstream table will extend this chain.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Launch document parsing
# MAGIC Parsing the in-scope documents with `ai_parse_document` takes a few minutes, so it runs in the background while we cover Phase 2.

# COMMAND ----------

launch_job("parse_documents", {}, run_name="meridian-parse-documents")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC | You did | Why it matters |
# MAGIC |---|---|
# MAGIC | Kept raw bytes in Bronze with a hash and timestamp | Re-parse without re-ingesting; trace any answer to its file |
# MAGIC | Checked coverage against the eval set | Found a document gap that no later phase could fix |
# MAGIC | Re-ran the same Auto Loader code | Only the new file was processed: incremental by design |
# MAGIC | Recorded scope decisions in a registry | Visible, auditable and reversible curation |

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT file_name,
# MAGIC        e.pos                               AS n,
# MAGIC        e.value:type::string                AS type,
# MAGIC        substr(e.value:content::string, 1, 250) AS content
# MAGIC FROM workspace.meridian_rag.silver_parsed,
# MAGIC      LATERAL variant_explode(parsed:document:elements) AS e
# MAGIC WHERE file_name LIKE '%AE_UAE%' OR file_name LIKE '%MAT-032_v2025%'
# MAGIC ORDER BY file_name, e.pos

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) AS parsed_files,
# MAGIC        count_if(parsed:error_status IS NOT NULL) AS with_errors
# MAGIC FROM workspace.meridian_rag.silver_parsed;
# MAGIC
# MAGIC SELECT count(*) AS html_files FROM workspace.meridian_rag.silver_parsed_html;