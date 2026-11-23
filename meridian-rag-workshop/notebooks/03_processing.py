# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 2: Data processing
# MAGIC
# MAGIC **Time box:** 12 minutes
# MAGIC
# MAGIC The parse job from Lab 1 ran `ai_parse_document` over every in-scope file. You'll compare its output with the naive
# MAGIC baseline, turn it into clean typed elements, extract each document's metadata and check it against ground truth,
# MAGIC find near-duplicates with MinHash, and filter out drafts and personal data. Then you'll launch a job that measures
# MAGIC what all of this was worth.
# MAGIC
# MAGIC **Done when:** `silver_documents` exists with 28 documents marked for indexing, and the follow-up job is running.

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Guard cell
# MAGIC Checks the parse job. If it failed or is still running, set `use_checkpoint=True` to continue from the reference parse.

# COMMAND ----------

import importlib, meridian_utils
importlib.reload(meridian_utils)        # always use the file on disk
from meridian_utils import *
self_check()                            # clear message if meridian_utils.py is out of date
import pandas as pd
cfg = guard_lab2(use_checkpoint=False)
PARSED = spark.table(cfg.table("silver_parsed"))
VARIANT_MODE = "parsed" in PARSED.columns          # real ai_parse_document output (vs checkpoint markdown)
print("Working from:", "ai_parse_document output" if VARIANT_MODE else "reference parse checkpoint")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Before and after
# MAGIC Here is what the baseline extracted from the scanned UAE addendum and the waiver matrix, next to what the parser returned.

# COMMAND ----------

naive = spark.table(cfg.table("naive_chunks")).select("file_name", "text").toPandas()
for f in ["MGB-KYC-ADD-AE", "MGB-TF-MAT-032_v2025.1"]:
    text = " ".join(naive[naive.file_name.str.startswith(f)].text)
    print(f"--- NAIVE  {f}: {len(text)} characters")
    print(text[:400] or "(nothing at all)", "\n")

# COMMAND ----------

if VARIANT_MODE:
    display(spark.sql(f"""
        SELECT file_name, e.pos AS n, e.value:type::string AS type, substr(e.value:content::string, 1, 300) AS content
        FROM {cfg.table('silver_parsed')}, LATERAL variant_explode(parsed:document:elements) AS e
        WHERE file_name LIKE 'MGB-KYC-ADD-AE%' OR file_name LIKE 'MGB-TF-MAT-032_v2025%'
        ORDER BY file_name, n"""))
else:
    display(PARSED.filter("file_name LIKE 'MGB-KYC-ADD-AE%' OR file_name LIKE 'MGB-TF-MAT-032_v2025%'"))

# COMMAND ----------

# MAGIC %md
# MAGIC **What changed:** the scanned page is now readable, including the two UAE rules the baseline missed. Every element
# MAGIC has a **type**: headers and footers are labelled, section headings are marked, and the waiver matrix is an HTML table
# MAGIC with its merged cells intact.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: One row per element
# MAGIC We explode every document into typed elements: the PDFs, images, DOCX and PPTX from the parser, and the HTML pages with a
# MAGIC small parser of our own (`ai_parse_document` doesn't take HTML) that produces the same element types.

# COMMAND ----------

rows, html_meta = [], {}
if VARIANT_MODE:
    exploded = spark.sql(f"""
        SELECT file_name, e.pos AS pos, e.value:type::string AS type, e.value:content::string AS content
        FROM {cfg.table('silver_parsed')}, LATERAL variant_explode(parsed:document:elements) AS e""").toPandas()
    rows.extend(exploded.to_dict("records"))
else:
    for r in PARSED.collect():
        for i, e in enumerate(markdown_to_elements(r.parsed_markdown)):
            rows.append({"file_name": r.file_name, "pos": i, **e})

in_scope_html = (spark.table(cfg.table("bronze_documents"))
                 .join(spark.table(cfg.table("source_registry")).filter("in_scope").select("file_name"), "file_name")
                 .filter("file_ext IN ('html', 'htm')").select("file_name", "content").collect())
for r in in_scope_html:
    els, meta = html_to_elements(bytes(r.content))
    html_meta[r.file_name] = meta
    rows.extend({"file_name": r.file_name, "pos": i, **e} for i, e in enumerate(els))

elements = pd.DataFrame(rows)
spark.createDataFrame(elements).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("silver_elements"))
ok(f"silver_elements: {len(elements)} elements from {elements.file_name.nunique()} files")
display(elements.groupby("type").size().reset_index(name="elements").sort_values("elements", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Clean
# MAGIC Running headers, footers and page numbers repeat on every page. In a chunk they're pure noise, and they make unrelated
# MAGIC chunks look similar. Because the parser labelled them, cleaning is a filter on type: no fragile text matching.
# MAGIC
# MAGIC Tables become markdown. Merged cells are repeated in every row they cover, so each row still makes sense on its own after chunking.

# COMMAND ----------

clean = elements[~elements.type.isin(NOISE_TYPES)].copy()
print(f"Removed {len(elements) - len(clean)} header, footer and page-number elements")
spark.createDataFrame(clean).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("silver_elements_clean"))

matrix = clean[clean.file_name.str.startswith("MGB-TF-MAT-032_v2025") & clean.type.isin(["table", "table_md"])]
print(elements_to_markdown(matrix.to_dict("records")[-1:]))

# COMMAND ----------

# MAGIC %md
# MAGIC Every row now carries "Applicant written consent required (a)", and the two header rows have become
# MAGIC "Approval authority / Primary approver". Compare that with the one-cell-per-line version the baseline embedded.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Extract metadata, then check it
# MAGIC Retrieval filters, version handling and citations all need metadata. Each PDF and DOCX has a cover block, which the parser
# MAGIC returned as the first table; HTML pages carry meta tags. Files without either fall back to what we can infer from the text.
# MAGIC We then validate every field against the ground-truth manifest.

# COMMAND ----------

bronze_hint = {r.file_name: r.doc_id_hint for r in spark.table(cfg.table("bronze_documents")).select("file_name", "doc_id_hint").collect()}
docs = []
for fname, g in elements.sort_values("pos").groupby("file_name"):
    tables = g[g.type == "table"]
    if fname in html_meta:
        meta, source = dict(html_meta[fname]), "html meta tags"
    elif len(tables) and parse_cover_table(tables.iloc[0].content).get("doc_id"):
        meta, source = normalise_meta(parse_cover_table(tables.iloc[0].content)), "cover block"
    else:
        text = " ".join(g.content.astype(str))
        m_id = re.search(r"Document ID:?\s*\|?\s*(MGB-[A-Z0-9-]+)", text) or re.search(r"\b(MGB-[A-Z]+(?:-[A-Z]+)?-[0-9A-Z]{2,3})\b", text)
        m_v = re.search(r"Version:?\s*\|?\s*(\d{4}\.\d+(?:-draft)?)", text) or re.search(r"\bv(\d{4}\.\d+)", text)
        m_s = re.search(r"Status:?\s*\|?\s*(\w+)", text)
        m_e = re.search(r"Effective:?\s*(?:date)?\s*\|?\s*(\d{4}-\d{2}-\d{2})", text)
        m_j = re.search(r"Jurisdiction:?\s*\|?\s*([A-Z]{2,6})", text)
        m_c = re.search(r"Classification:?\s*\|?\s*(restricted[ _]pii|confidential|internal)", text, re.I)
        meta = normalise_meta({"doc_id": m_id.group(1) if m_id else bronze_hint.get(fname),
                               "version": m_v.group(1) if m_v else None, "status": m_s.group(1) if m_s else "current",
                               "effective_date": m_e.group(1) if m_e else None,
                               "jurisdiction": m_j.group(1) if m_j else "GLOBAL",
                               "sensitivity": m_c.group(1) if m_c else "internal"})
        source = "inferred from text"
    meta["doc_id"] = meta.get("doc_id") or bronze_hint.get(fname)
    titles = g[g.type == "title"].content
    if len(titles):
        title = titles.iloc[0]
    else:   # some DOCX files have no element typed 'title': use the first short heading or line before the cover block
        first_table = g[g.type == "table"].pos.min() if len(tables) else g.pos.max() + 1
        cands = g[(g.pos < first_table) & g.type.isin(["section_header", "text", "page_header"])].content.astype(str)
        cands = [c.strip() for c in cands if 3 < len(c.strip()) < 90 and "Meridian Global Bank" not in c]
        title = cands[0] if cands else fname
    docs.append({"file_name": fname, "title": title, "meta_source": source,
                 **{k: meta.get(k) for k in ["doc_id", "version", "status", "effective_date", "superseded_date",
                                              "owner", "jurisdiction", "sensitivity"]},
                 **derive_from_doc_id(meta.get("doc_id"))})
docs = pd.DataFrame(docs)
docs["status"] = docs.status.fillna("current"); docs["jurisdiction"] = docs.jurisdiction.fillna("GLOBAL")
docs["sensitivity"] = docs.sensitivity.fillna("internal")
display(docs[["file_name", "doc_id", "version", "status", "jurisdiction", "sensitivity", "doc_type", "authority_rank", "meta_source"]])

# COMMAND ----------

truth = pd.read_csv(f"{cfg.work_path}/reference/corpus_manifest.csv", dtype=str)
truth = truth.rename(columns={"effective_date": "effective_date_t"}).set_index("file_name")
fields = {"doc_id": "doc_id", "version": "version", "status": "status", "jurisdiction": "jurisdiction",
          "effective_date": "effective_date_t", "sensitivity": "sensitivity"}
check = docs.set_index("file_name").join(truth[[v for v in fields.values()]], rsuffix="_truth", how="left")
summary, misses = [], []
for f, t in fields.items():
    tcol = t if t != f else f + "_truth"
    eq = check[f].astype(str).str.lower() == check[tcol].astype(str).str.lower()
    summary.append({"field": f, "match_rate": f"{eq.mean():.0%}"})
    misses += [{"file_name": i, "field": f, "extracted": check.loc[i, f], "truth": check.loc[i, tcol]} for i in check.index[~eq]]
display(pd.DataFrame(summary))
display(pd.DataFrame(misses)) if misses else ok("Every extracted field matches the manifest")

# COMMAND ----------

# MAGIC %md
# MAGIC **Look at the "FINAL copy" file.** Its name carries no doc ID, but its cover block says `MGB-KYC-POL-001`, version `2025.1`.
# MAGIC Metadata from the document itself beats metadata from the file name. Any mismatches above come from files without a
# MAGIC cover block (the slide deck and the photographed card), where we had to infer from the text: a realistic reminder that
# MAGIC extracted metadata needs validating.

# COMMAND ----------

# MAGIC %md
# MAGIC The cover block has done its job: its fields are now columns, and Phase 3 adds them to each chunk as a short context line.
# MAGIC Left in the text, it would repeat the same boilerplate at the top of every document. We drop it from the clean elements.

# COMMAND ----------

def is_cover(e):
    c = str(e["content"] or "")
    return (e["type"] == "table" and bool(parse_cover_table(c).get("doc_id"))) or \
           (e["type"] == "text" and c.startswith("Document ID:"))

before = len(clean)
clean = clean[~clean.apply(is_cover, axis=1)].copy()
spark.createDataFrame(clean).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("silver_elements_clean"))
ok(f"Removed {before - len(clean)} cover blocks from the document text")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Find near-duplicates with MinHash
# MAGIC MinHash estimates how much two documents' text overlaps (Jaccard similarity on 5-word shingles) without comparing every
# MAGIC word. Watch two things: the emailed copy, and the 2024 and 2025 versions of the same policy.

# COMMAND ----------

docs["doc_markdown"] = [elements_to_markdown(clean[clean.file_name == f].sort_values("pos").to_dict("records")) for f in docs.file_name]
sigs = {f: minhash_signature(shingles(t)) for f, t in zip(docs.file_name, docs.doc_markdown)}
pairs = []
names = list(sigs)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        s = minhash_similarity(sigs[names[i]], sigs[names[j]])
        if s >= 0.4:
            pairs.append({"file_a": names[i], "file_b": names[j], "similarity": round(s, 2)})
display(pd.DataFrame(pairs).sort_values("similarity", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC **Similar isn't the same as duplicate.** The emailed copy scores close to 1.0 with the 2025 policy: a true duplicate.
# MAGIC The 2024 and 2025 versions are also very similar, but they're different versions with different rules, and Q15 needs the
# MAGIC old one. So we only treat a pair as duplicates when the text is near-identical **and** the doc ID and version match.
# MAGIC We keep the file whose name follows the convention.

# COMMAND ----------

key = docs.set_index("file_name")[["doc_id", "version"]]
dupe_of = {}
for p in pairs:
    a, b = p["file_a"], p["file_b"]
    if p["similarity"] >= 0.9 and tuple(key.loc[a]) == tuple(key.loc[b]):
        keep, drop = (a, b) if a.startswith(str(key.loc[a, "doc_id"])) else (b, a)
        dupe_of[drop] = keep
for d, k in dupe_of.items():
    ok(f"{d} is a duplicate of {k}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Filter, and write silver_documents
# MAGIC Drafts are never authoritative. Files classified as containing personal data never enter a policy copilot's index.
# MAGIC Superseded documents stay in, tagged, so historical questions can still be answered; Phase 7 filters them by default.

# COMMAND ----------

def exclusion(r):
    if r.file_name in dupe_of: return f"duplicate of {dupe_of[r.file_name]}"
    if r.status == "draft": return "draft: not in force"
    if r.sensitivity == "restricted_pii": return "personal data"
    return None

docs["exclusion_reason"] = docs.apply(exclusion, axis=1)
docs["include_in_index"] = docs.exclusion_reason.isna()
docs["authority_rank"] = docs.authority_rank.astype(int)
spark.createDataFrame(docs).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("silver_documents"))

n_bronze = spark.table(cfg.table("bronze_documents")).count()
n_scope = spark.table(cfg.table("source_registry")).filter("in_scope").count()
display(pd.DataFrame([
    {"stage": "Landed in Bronze", "documents": n_bronze},
    {"stage": "In scope (Lab 1 registry)", "documents": n_scope},
    {"stage": "Parsed with metadata", "documents": len(docs)},
    {"stage": "Marked for indexing", "documents": int(docs.include_in_index.sum())},
]))
display(docs[~docs.include_in_index][["file_name", "exclusion_reason"]])

# COMMAND ----------

# MAGIC %md
# MAGIC **Quick check:** is the UAE rule the baseline missed now in the text we'll index?

# COMMAND ----------

uae = docs[docs.doc_id == "MGB-KYC-ADD-AE"].doc_markdown.iloc[0]
print("Six-monthly review present:", "every 6 months" in uae)
print(uae[uae.find("## 3."):][:400])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Launch the follow-up job
# MAGIC It builds three chunking variants for Phase 3, embeds the clean documents with the **same** fixed-size chunking as the
# MAGIC baseline, and evaluates the result as run `after_p2`. Because only Phase 2 changed, the difference from the baseline is
# MAGIC what parsing, cleaning and filtering were worth. It runs while we cover Phase 3.

# COMMAND ----------

launch_job("phase2_followup", {}, run_name="meridian-phase2-followup")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC | You did | Why it matters |
# MAGIC |---|---|
# MAGIC | Parsed with `ai_parse_document` | Scanned pages and images became text; tables kept their structure |
# MAGIC | Exploded into typed elements | Cleaning became a filter, and Phase 3 can chunk along section boundaries |
# MAGIC | Expanded merged table cells | Every table row reads on its own once chunked |
# MAGIC | Extracted and validated metadata | Filters, versions and citations depend on it, so it has to be right |
# MAGIC | Told duplicates from versions | Removed the copy without losing the history Q15 needs |
# MAGIC | Filtered drafts and personal data | The 9-month draft and the passport number can no longer be retrieved |
