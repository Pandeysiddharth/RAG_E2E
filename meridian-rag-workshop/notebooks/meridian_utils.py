"""
Meridian RAG workshop: shared helpers.

Every notebook and background job imports this module, so the configuration, the baseline pipeline
and the scorers are identical everywhere. Participants are welcome to read it, but don't need to edit it.
"""
from __future__ import annotations

import io, json, os, re, shutil, time
from dataclasses import dataclass, asdict

import numpy as np

# ----------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------
CATALOG = os.environ.get("MERIDIAN_CATALOG", "workspace")   # Free Edition default catalog
SCHEMA = os.environ.get("MERIDIAN_SCHEMA", "meridian_rag")
RAW_VOLUME = "raw"            # corpus files land here
WORK_VOLUME = "workshop"      # config, eval set, checkpoints, stream checkpoints
AI_SEARCH_ENDPOINT = "meridian-ai-search"

CHAT_PREFS = [
    "databricks-claude-sonnet-4-5", "databricks-claude-3-7-sonnet", "databricks-meta-llama-3-3-70b-instruct",
    "databricks-llama-4-maverick", "databricks-gpt-oss-120b", "databricks-gpt-oss-20b",
]
EMBED_PREFS = ["databricks-gte-large-en", "databricks-bge-large-en", "databricks-qwen3-embedding-0-6b"]


@dataclass
class Cfg:
    catalog: str = CATALOG
    schema: str = SCHEMA
    user: str = ""
    chat_endpoint: str = ""
    embed_endpoint: str = ""
    judge_model: str = ""          # empty = Databricks-hosted judges
    ai_search_endpoint: str = AI_SEARCH_ENDPOINT
    embed_batch: int = 4           # tuned by diagnose_embeddings() in 00_setup
    embed_pause: float = 0.0       # seconds between embedding calls, if the rate limit needs it
    experiment_path: str = ""
    bundle_root: str = ""

    @property
    def fq(self): return f"{self.catalog}.{self.schema}"
    @property
    def raw_path(self): return f"/Volumes/{self.catalog}/{self.schema}/{RAW_VOLUME}"
    @property
    def work_path(self): return f"/Volumes/{self.catalog}/{self.schema}/{WORK_VOLUME}"
    @property
    def config_file(self): return f"{self.work_path}/config.json"
    def table(self, name): return f"{self.fq}.{name}"


def _spark():
    from pyspark.sql import SparkSession
    return SparkSession.getActiveSession()


def load_cfg() -> Cfg:
    """Read the configuration written by 00_setup."""
    cfg = Cfg()
    try:
        with open(cfg.config_file) as fh:
            cfg = Cfg(**json.load(fh))
    except FileNotFoundError:
        raise RuntimeError("No workshop config found. Run 00_setup first, or ask a helper.") from None
    return cfg


def save_cfg(cfg: Cfg):
    with open(cfg.config_file, "w") as fh:
        json.dump(asdict(cfg), fh, indent=2)


def ok(msg): print(f"  \u2705 {msg}")
def warn(msg): print(f"  \u26a0\ufe0f  {msg}")
def fail(msg): print(f"  \u274c {msg}")

# ----------------------------------------------------------------------------------------------
# Model endpoints (Foundation Model APIs, pay-per-token)
# ----------------------------------------------------------------------------------------------
_RETRYABLE = {429, 500, 502, 503, 504}


def _invoke(endpoint: str, body: dict, timeout=60, max_attempts=6, quiet=False):
    """Call a serving endpoint directly, with visible, bounded retries.

    We don't use a client library's built-in retries here because they retry silently for up to 10 minutes.
    On Free Edition's rate limits you want to see what's happening and fail fast with a clear message."""
    import requests
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()
    url = f"{w.config.host.rstrip('/')}/serving-endpoints/{endpoint}/invocations"
    last = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.post(url, headers={**w.config.authenticate(), "Content-Type": "application/json"},
                              json=body, timeout=timeout)
        except requests.exceptions.Timeout:
            last = f"no response within {timeout}s"
            r = None
        if r is not None and r.status_code == 200:
            return r.json()
        if r is not None and r.status_code not in _RETRYABLE:
            raise RuntimeError(f"{endpoint} returned HTTP {r.status_code}: {r.text[:300]}")
        if r is not None:
            last = f"HTTP {r.status_code}" + (" (rate limited)" if r.status_code == 429 else "")
            wait = float(r.headers.get("Retry-After") or min(2 ** attempt, 30))
        else:
            wait = min(2 ** attempt, 30)
        if not quiet:
            print(f"    {endpoint}: {last}, retry {attempt}/{max_attempts - 1} in {wait:.0f}s")
        if attempt < max_attempts:
            time.sleep(wait)
    raise RuntimeError(f"{endpoint} failed after {max_attempts} attempts ({last}). "
                       "If this says 'rate limited', wait a minute and re-run, or lower cfg.embed_batch.")


def chat(messages, cfg: Cfg, max_tokens=500, temperature=0.0):
    resp = _invoke(cfg.chat_endpoint, {"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
                   timeout=120)
    return resp["choices"][0]["message"]["content"], resp.get("usage", {})


# ---- embedding cache: every text is embedded at most once per model, across all labs and re-runs ----
_EMB_CACHE = {}


def _cache_file(cfg: Cfg):
    return f"{cfg.work_path}/cache/emb_{cfg.embed_endpoint}.jsonl"


def _load_cache(cfg: Cfg):
    key = cfg.embed_endpoint
    if key not in _EMB_CACHE:
        _EMB_CACHE[key] = {}
        try:
            with open(_cache_file(cfg)) as fh:
                for line in fh:
                    rec = json.loads(line); _EMB_CACHE[key][rec["h"]] = rec["v"]
        except FileNotFoundError:
            pass
    return _EMB_CACHE[key]


def _text_hash(t: str):
    import hashlib
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def embed(texts, cfg: Cfg, batch=None, progress=None):
    """Embed texts with caching, adaptive batch size and visible progress."""
    texts = list(texts)
    cache = _load_cache(cfg)
    hashes = [_text_hash(t) for t in texts]
    todo = [i for i, h in enumerate(hashes) if h not in cache]
    progress = len(texts) > 8 if progress is None else progress
    if progress and len(todo) < len(texts):
        print(f"    {len(texts) - len(todo)} of {len(texts)} already cached")
    size = max(1, batch or cfg.embed_batch or 4)
    os.makedirs(os.path.dirname(_cache_file(cfg)), exist_ok=True)
    done, i, t0 = 0, 0, time.time()
    with open(_cache_file(cfg), "a") as fh:
        while i < len(todo):
            idx = todo[i:i + size]
            try:
                resp = _invoke(cfg.embed_endpoint, {"input": [texts[j] for j in idx]}, max_attempts=3, quiet=size > 1)
            except RuntimeError:
                if size > 1:                       # back off to a smaller batch rather than failing
                    size = max(1, size // 2)
                    print(f"    lowering embedding batch size to {size}")
                    continue
                raise
            for j, d in zip(idx, resp["data"]):
                cache[hashes[j]] = d["embedding"]
                fh.write(json.dumps({"h": hashes[j], "v": d["embedding"]}) + "\n")
            i += len(idx); done += len(idx)
            if progress:
                print(f"    embedded {done}/{len(todo)}  ({time.time() - t0:.0f}s)")
            if cfg.embed_pause:
                time.sleep(cfg.embed_pause)
    return [cache[h] for h in hashes]


def diagnose_embeddings(cfg: Cfg, sizes=(1, 2, 4, 8, 16)):
    """Find the largest batch size this workspace's embedding endpoint accepts without being rate limited."""
    sample = ["Meridian Global Bank onboarding policy sample text number %d. " % k * 12 for k in range(max(sizes))]
    best = 1
    print(f"  Testing {cfg.embed_endpoint}:")
    for n in sizes:
        t0 = time.time()
        try:
            _invoke(cfg.embed_endpoint, {"input": sample[:n]}, max_attempts=1, quiet=True, timeout=45)
            print(f"    batch {n:>2}: ok in {time.time() - t0:.1f}s")
            best = n
        except RuntimeError as e:
            print(f"    batch {n:>2}: failed ({str(e)[:80]})")
            break
        time.sleep(1)
    cfg.embed_batch = max(1, best // 2) if best >= 4 else best   # leave headroom
    print(f"  Using batch size {cfg.embed_batch}")
    return cfg.embed_batch


# ----------------------------------------------------------------------------------------------
# File naming helpers
# ----------------------------------------------------------------------------------------------
def doc_key_from_filename(name: str) -> str:
    """'MGB-TF-MAT-032_v2025.1_Waiver...pdf' -> 'MGB-TF-MAT-032@2025.1'; unknown names -> stem."""
    stem = os.path.splitext(os.path.basename(name))[0]
    first = stem.split("_")[0]
    if not first.startswith("MGB-"):
        return stem
    m = re.search(r"_v(\d{4}\.\d+)_", stem + "_")
    return f"{first}@{m.group(1)}" if m else first


def base_doc_id(key: str) -> str:
    return key.split("@")[0]

# ----------------------------------------------------------------------------------------------
# NAIVE baseline: plain extraction + fixed-size chunks (deliberately weak, fixed in Phases 2 to 4)
# ----------------------------------------------------------------------------------------------
def extract_text_naive(name: str, data: bytes) -> str:
    ext = name.lower().rsplit(".", 1)[-1]
    try:
        if ext == "pdf":
            from pypdf import PdfReader
            return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
        if ext == "docx":
            import docx
            return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)   # tables ignored
        if ext == "pptx":
            from pptx import Presentation
            return "\n".join(sh.text_frame.text for s in Presentation(io.BytesIO(data)).slides
                             for sh in s.shapes if sh.has_text_frame)                    # tables and notes ignored
        if ext in ("html", "htm"):
            from bs4 import BeautifulSoup
            return BeautifulSoup(data, "html.parser").get_text("\n")                     # nav and footer kept
        return ""                                                                         # images: nothing
    except Exception as e:  # noqa: BLE001
        return ""


def chunk_fixed(text: str, size=1000):
    text = text or ""
    return [text[i:i + size] for i in range(0, len(text), size) if text[i:i + size].strip()]


def naive_chunk_rows(files):
    """files: iterable of (file_name, bytes) -> list of chunk dicts."""
    rows = []
    for name, data in files:
        key = doc_key_from_filename(name)
        for j, c in enumerate(chunk_fixed(extract_text_naive(name, data))):
            rows.append({"chunk_id": f"{os.path.basename(name)}::{j}", "file_name": os.path.basename(name),
                         "doc_key": key, "chunk_index": j, "text": c})
    return rows

# ----------------------------------------------------------------------------------------------
# Baseline RAG (in-memory vector search; the real AI Search index arrives in Phase 5)
# ----------------------------------------------------------------------------------------------
NAIVE_SYSTEM = "You are a helpful assistant. Answer the question using the context."


class NaiveRAG:
    def __init__(self, cfg: Cfg, table="naive_chunks", k=5):
        import mlflow
        from mlflow.entities import SpanType
        self.cfg, self.k = cfg, k
        pdf = _spark().table(cfg.table(table)).toPandas()
        self.chunks = pdf
        m = np.array(pdf["embedding"].tolist(), dtype=np.float32)
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)
        # decorate here so the module imports cleanly even without mlflow
        self.retrieve = mlflow.trace(span_type=SpanType.RETRIEVER, name="retrieve")(self._retrieve)
        self.build_prompt = mlflow.trace(span_type=SpanType.PARSER, name="build_prompt")(self._build_prompt)
        self.generate = mlflow.trace(span_type=SpanType.CHAT_MODEL, name="chat_completion")(self._generate)
        self.answer = mlflow.trace(span_type=SpanType.CHAIN, name="naive_rag")(self._answer)

    def _retrieve(self, question: str):
        from mlflow.entities import Document
        q = np.array(embed([question], self.cfg)[0], dtype=np.float32)
        q /= np.linalg.norm(q)
        idx = np.argsort(-self.matrix @ q)[: self.k]
        return [Document(page_content=self.chunks.iloc[i]["text"], id=self.chunks.iloc[i]["chunk_id"],
                         metadata={"doc_uri": self.chunks.iloc[i]["doc_key"], "chunk_id": self.chunks.iloc[i]["chunk_id"]})
                for i in idx]

    def _build_prompt(self, question, docs):
        context = "\n\n".join(d.page_content for d in docs)           # no source IDs, no versions: deliberately naive
        return [{"role": "system", "content": NAIVE_SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]

    def _generate(self, messages):
        text, usage = chat(messages, self.cfg)
        import mlflow
        span = mlflow.get_current_active_span()
        if span and usage:
            span.set_attributes({"usage.prompt_tokens": usage.get("prompt_tokens"),
                                 "usage.completion_tokens": usage.get("completion_tokens")})
        return text

    def _answer(self, question: str) -> str:
        docs = self.retrieve(question)
        return self.generate(self.build_prompt(question, docs))


PIPELINES = {"naive": lambda cfg: NaiveRAG(cfg)}

# ----------------------------------------------------------------------------------------------
# Evaluation set and scorers
# ----------------------------------------------------------------------------------------------
def load_eval_records(cfg: Cfg):
    """Returns records in mlflow.genai.evaluate format: [{'inputs': {...}, 'expectations': {...}}]."""
    recs = []
    with open(f"{cfg.work_path}/eval/meridian_eval_set.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            recs.append({"inputs": r["inputs"], "expectations": r["expectations"]})
    return recs


REFUSAL = ["i don't know", "i do not know", "cannot", "can't", "unable", "not able", "outside", "out of scope",
           "not covered", "no information", "don't have", "do not have", "not provide", "not in the"]


def get_scorers(cfg: Cfg):
    from mlflow.entities import Feedback, SpanType
    from mlflow.genai.scorers import (Correctness, RelevanceToQuery, RetrievalGroundedness,
                                      RetrievalSufficiency, Safety, scorer)

    @scorer
    def recall_at_5(trace, expectations):
        expected = set(expectations.get("expected_doc_ids") or [])
        if not expected:
            return None                                   # abstain questions: not applicable
        got = []
        for sp in trace.search_spans(span_type=SpanType.RETRIEVER):
            for d in (sp.outputs or [])[:5]:
                md = d.get("metadata", {}) if isinstance(d, dict) else getattr(d, "metadata", {})
                got.append(md.get("doc_uri", ""))
        hit = len(expected & set(got)) / len(expected)
        return Feedback(value=hit, rationale=f"expected {sorted(expected)}; retrieved {got}")

    @scorer
    def abstention_check(outputs, expectations):
        mode = expectations.get("should_abstain", "none")
        text = str(outputs).lower()
        refused = any(m in text for m in REFUSAL)
        if mode == "full":
            return Feedback(value=refused, rationale="should decline")
        if mode == "deflect":
            return Feedback(value=("approver" in text or "decision" in text), rationale="should leave decision to approver")
        return Feedback(value=not refused, rationale="should answer")

    @scorer
    def forbidden_content(outputs, expectations):
        bad = [f for f in (expectations.get("forbidden_content") or []) if f.lower() in str(outputs).lower()]
        return Feedback(value=not bad, rationale=f"forbidden strings found: {bad}" if bad else "clean")

    jm = {"model": cfg.judge_model} if cfg.judge_model else {}
    return [Correctness(**jm), RetrievalSufficiency(**jm), RetrievalGroundedness(**jm), RelevanceToQuery(**jm),
            Safety(**jm), recall_at_5, abstention_check, forbidden_content]


def record_scorecard(cfg: Cfg, run_name: str, metrics: dict):
    rows = [(run_name, k, float(v)) for k, v in metrics.items() if isinstance(v, (int, float))]
    sp = _spark()
    df = sp.createDataFrame(rows, "run_name string, metric string, value double")
    from pyspark.sql import functions as F
    df.withColumn("logged_at", F.current_timestamp()).write.mode("append").saveAsTable(cfg.table("scorecard"))


def show_scorecard(cfg: Cfg):
    return _spark().sql(f"""
        SELECT run_name, metric, round(value, 3) AS value, logged_at FROM {cfg.table('scorecard')}
        WHERE metric RLIKE '(mean|ratio)$' ORDER BY logged_at, metric""")

# ----------------------------------------------------------------------------------------------
# Background jobs (serverless) so long steps run while the next deck chapter is presented
# ----------------------------------------------------------------------------------------------
def current_notebook_dir():
    from pyspark.dbutils import DBUtils
    path = DBUtils(_spark()).notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    return os.path.dirname(path)


def launch_job(notebook_name: str, params: dict, run_name: str):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service import jobs
    w = WorkspaceClient()
    nb = f"{current_notebook_dir()}/jobs/{notebook_name}"
    run = w.jobs.submit(run_name=run_name, tasks=[jobs.SubmitTask(
        task_key="main", notebook_task=jobs.NotebookTask(notebook_path=nb, base_parameters=params))])
    run_id = run.response.run_id if hasattr(run, "response") else run.run_id
    host = w.config.host.rstrip("/")
    print(f"\U0001F680 Launched '{run_name}' (run {run_id}). It keeps running while we cover the next phase.")
    print(f"   Watch it here: {host}/#job/runs/{run_id}  or under Jobs & Pipelines > Runs")
    with open(f"{load_cfg().work_path}/last_job_{notebook_name}.json", "w") as fh:
        json.dump({"run_id": run_id, "run_name": run_name, "launched": time.time()}, fh)
    return run_id


def job_status(notebook_name: str):
    """Returns (state, result) for the most recent launch of a job notebook, or (None, None)."""
    from databricks.sdk import WorkspaceClient
    try:
        with open(f"{load_cfg().work_path}/last_job_{notebook_name}.json") as fh:
            run_id = json.load(fh)["run_id"]
    except FileNotFoundError:
        return None, None
    r = WorkspaceClient().jobs.get_run(run_id)
    st = r.state
    life = getattr(st.life_cycle_state, "value", st.life_cycle_state) if st else None
    res = getattr(st.result_state, "value", st.result_state) if st and st.result_state else None
    return life, res

# ----------------------------------------------------------------------------------------------
# Guards: every lab starts with one, so a slow job or a missed step never blocks anyone
# ----------------------------------------------------------------------------------------------
def table_exists(cfg, name):
    return _spark().catalog.tableExists(cfg.table(name))


def build_naive_chunks(cfg: Cfg, use_checkpoint=False):
    import pandas as pd
    sp = _spark()
    ckpt = f"{cfg.work_path}/checkpoints"
    pre = f"{ckpt}/cp_00_naive_chunks_embedded.jsonl"
    # 1. Pre-embedded checkpoint shipped by the instructor: no model calls at all
    if os.path.exists(pre):
        pdf = pd.read_json(pre, lines=True)
        if pdf["embed_model"].iloc[0] == cfg.embed_endpoint:
            pdf = pdf.drop(columns=["embed_model"])
            sp.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("naive_chunks"))
            ok(f"{cfg.table('naive_chunks')} loaded with pre-computed embeddings ({len(pdf)} chunks, no model calls needed)")
            return
        warn(f"Pre-computed embeddings were made with {pdf['embed_model'].iloc[0]}, but you have {cfg.embed_endpoint}. Embedding live.")
    # 2. Chunks: live extraction, falling back to the checkpoint only if extraction itself fails
    if use_checkpoint:
        pdf = pd.read_json(f"{ckpt}/cp_00_naive_chunks.jsonl", lines=True); print("  Loaded naive chunks from checkpoint cp_00.")
    else:
        try:
            files = [(f, open(f"{cfg.raw_path}/{f}", "rb").read()) for f in sorted(os.listdir(cfg.raw_path))
                     if not f.startswith("MGB-KYC-ADD-UK")]          # the UK addendum arrives in Lab 1
            pdf = pd.DataFrame(naive_chunk_rows(files))
        except Exception as e:  # noqa: BLE001
            warn(f"Live extraction failed ({e}); using checkpoint cp_00 chunks")
            pdf = pd.read_json(f"{ckpt}/cp_00_naive_chunks.jsonl", lines=True)
    # 3. Embeddings: cached, adaptive, with progress. Re-running resumes where it stopped.
    print(f"  Embedding {len(pdf)} chunks with {cfg.embed_endpoint} (batch size {cfg.embed_batch}) ...")
    pdf["embedding"] = embed(pdf["text"].tolist(), cfg, progress=True)
    sp.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("naive_chunks"))
    ok(f"{cfg.table('naive_chunks')} ready ({len(pdf)} chunks)")


def export_embedded_checkpoint(cfg: Cfg, bundle_checkpoints_dir: str):
    """Instructor helper: after a successful setup, write naive chunks with embeddings into the bundle,
    so participants' setup needs no embedding calls at all."""
    pdf = _spark().table(cfg.table("naive_chunks")).toPandas()
    pdf["embedding"] = pdf["embedding"].apply(list)
    pdf["embed_model"] = cfg.embed_endpoint
    path = f"{bundle_checkpoints_dir}/cp_00_naive_chunks_embedded.jsonl"
    pdf.to_json(path, orient="records", lines=True)
    ok(f"Wrote {len(pdf)} embedded chunks to {path}. Commit it to the workshop repo.")


def guard_lab0(use_checkpoint=False):
    print("Guard check for Lab 0")
    cfg = load_cfg(); ok(f"config loaded (chat: {cfg.chat_endpoint}, embeddings: {cfg.embed_endpoint})")
    if use_checkpoint or not table_exists(cfg, "naive_chunks"):
        warn("naive_chunks missing or checkpoint requested: rebuilding (cached embeddings are reused)")
        build_naive_chunks(cfg, use_checkpoint=True)
    else:
        n = _spark().table(cfg.table("naive_chunks")).count(); ok(f"naive_chunks has {n} chunks")
    import mlflow
    mlflow.set_experiment(cfg.experiment_path); ok(f"MLflow experiment: {cfg.experiment_path}")
    return cfg


def guard_lab1():
    print("Guard check for Lab 1")
    cfg = load_cfg(); ok("config loaded")
    life, res = job_status("run_evaluation")
    if life is None:
        warn("No baseline evaluation job found. Launch it from 01_baseline (last cell), then carry on here.")
    elif res == "SUCCESS":
        ok("Baseline evaluation finished. Your numbers are in the scorecard table.")
    elif life in ("PENDING", "RUNNING", "QUEUED", "BLOCKED"):
        warn(f"Baseline evaluation still {life.lower()}. That's fine, carry on: it doesn't block this lab.")
    else:
        fail(f"Baseline evaluation ended with {res or life}. Ask a helper; it doesn't block this lab.")
    import mlflow
    mlflow.set_experiment(cfg.experiment_path)
    return cfg
