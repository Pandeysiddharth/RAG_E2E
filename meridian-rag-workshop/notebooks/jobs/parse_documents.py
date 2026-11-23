# Databricks notebook source
# MAGIC %md
# MAGIC # Background job: parse documents with `ai_parse_document`
# MAGIC Launched at the end of Lab 1. It reads the in-scope Bronze files and writes `silver_parsed`, which Lab 2 builds on.
# MAGIC
# MAGIC `ai_parse_document` handles PDF, images, DOCX and PPTX, including scanned pages and tables. HTML isn't a supported
# MAGIC input, so HTML files are converted to text here. If the AI function isn't available in your workspace, the job falls
# MAGIC back to the reference parse shipped in checkpoint cp_02, so Lab 2 always has data.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements.txt
# MAGIC %restart_python

# COMMAND ----------

import os, sys
sys.path.append(os.path.abspath(".."))
from meridian_utils import *
from pyspark.sql import functions as F
import pandas as pd

cfg = load_cfg()
bronze = spark.table(cfg.table("bronze_documents")).join(
    spark.table(cfg.table("source_registry")).filter("in_scope").select("file_name"), "file_name")
binary = bronze.filter(~F.col("file_ext").isin("html", "htm"))
html = bronze.filter(F.col("file_ext").isin("html", "htm"))
print(f"{binary.count()} files for ai_parse_document, {html.count()} HTML files")

# COMMAND ----------

try:
    parsed = (binary
              .withColumn("parsed", F.expr("ai_parse_document(content, map('version', '2.0'))"))
              .select("file_name", "doc_id_hint", "file_ext", "parsed", F.lit("ai_parse_document").alias("parse_source")))
    parsed.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("silver_parsed"))
    ok(f"ai_parse_document parsed {spark.table(cfg.table('silver_parsed')).count()} files into silver_parsed")
except Exception as e:
    warn(f"ai_parse_document unavailable or failed ({str(e)[:200]}). Falling back to the reference parse in cp_02.")
    ref = pd.read_json(f"{cfg.work_path}/checkpoints/cp_02_reference_parse.jsonl", lines=True)
    keep = {r.file_name for r in bronze.select("file_name").collect()}
    ref = ref[ref.file_name.isin(keep) & ~ref.file_name.str.endswith(".html")]
    (spark.createDataFrame(ref[["file_name", "parsed_markdown", "parse_source"]])
          .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("silver_parsed")))

# COMMAND ----------

from bs4 import BeautifulSoup
rows = []
for r in html.select("file_name", "doc_id_hint", "content").collect():
    soup = BeautifulSoup(bytes(r.content), "html.parser")
    meta = {m["name"]: m["content"] for m in soup.find_all("meta") if m.get("name", "").startswith("mgb:")}
    for tag in soup(["nav", "footer", "style", "script"]):
        tag.decompose()
    rows.append({"file_name": r.file_name, "doc_id_hint": r.doc_id_hint, "html_meta": json.dumps(meta),
                 "parsed_text": soup.get_text("\n", strip=True)})
spark.createDataFrame(pd.DataFrame(rows)).write.mode("overwrite").option("overwriteSchema", "true") \
     .saveAsTable(cfg.table("silver_parsed_html"))
ok(f"{len(rows)} HTML files written to silver_parsed_html")
