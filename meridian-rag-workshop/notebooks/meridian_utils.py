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
def _deploy_client():
    from mlflow.deployments import get_deploy_client
    return get_deploy_client("databricks")


def detect_endpoints():
    """Pick the first available chat and embedding endpoints from the preference lists.
    Free Edition offers a subset of models, so we detect rather than hard-code."""
    from databricks.sdk import WorkspaceClient
    eps = {e.name: (e.task or "") for e in WorkspaceClient().serving_endpoints.list()}
    chat = next((n for n in CHAT_PREFS if n in eps), None) or next((n for n, t in eps.items() if "chat" in t), None)
    emb = next((n for n in EMBED_PREFS if n in eps), None) or next((n for n, t in eps.items() if "embedding" in t), None)
    return chat, emb, eps


def chat(messages, cfg: Cfg, max_tokens=500, temperature=0.0):
    resp = _deploy_client().predict(endpoint=cfg.chat_endpoint,
                                    inputs={"messages": messages, "max_tokens": max_tokens, "temperature": temperature})
    return resp["choices"][0]["message"]["content"], resp.get("usage", {})


def embed(texts, cfg: Cfg, batch=16):
    out = []
    client = _deploy_client()
    for i in range(0, len(texts), batch):
        resp = client.predict(endpoint=cfg.embed_endpoint, inputs={"input": texts[i:i + batch]})
        out.extend(d["embedding"] for d in resp["data"])
    return out

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
    if use_checkpoint:
        pdf = pd.read_json(f"{cfg.work_path}/checkpoints/cp_00_naive_chunks.jsonl", lines=True)
        print("  Loaded naive chunks from checkpoint cp_00.")
    else:
        files = [(f, open(f"{cfg.raw_path}/{f}", "rb").read()) for f in sorted(os.listdir(cfg.raw_path))]
        pdf = pd.DataFrame(naive_chunk_rows(files))
    print(f"  Embedding {len(pdf)} chunks with {cfg.embed_endpoint} ...")
    pdf["embedding"] = embed(pdf["text"].tolist(), cfg)
    sp.createDataFrame(pdf).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.table("naive_chunks"))
    ok(f"{cfg.table('naive_chunks')} ready ({len(pdf)} chunks)")


def guard_lab0(use_checkpoint=False):
    print("Guard check for Lab 0")
    cfg = load_cfg(); ok(f"config loaded (chat: {cfg.chat_endpoint}, embeddings: {cfg.embed_endpoint})")
    if use_checkpoint or not table_exists(cfg, "naive_chunks"):
        warn("naive_chunks missing or checkpoint requested: rebuilding")
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
