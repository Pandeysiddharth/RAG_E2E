# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Lab 10: Post-processing
# MAGIC
# MAGIC **Time box:** 8 minutes
# MAGIC
# MAGIC The augmented prompt tells the model the rules, but an instruction isn't enforcement: in the dry run, Q38's answer still
# MAGIC cited the stale FAQ. You'll add checks that run on every answer **before it's sent**: citations, source authority,
# MAGIC known stale facts, groundedness and personal data. An answer is released, repaired once, or replaced.
# MAGIC
# MAGIC **Done when:** you've seen each check fire and the `after_p10` evaluation is running.

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Guard cell

# COMMAND ----------

import importlib, meridian_utils
importlib.reload(meridian_utils)        # always use the file on disk
from meridian_utils import *
self_check()                            # clear message if meridian_utils.py is out of date
import pandas as pd
cfg, index = guard_lab10()
EVAL = {json.loads(l)["tags"]["question_id"]: json.loads(l) for l in open(f"{cfg.work_path}/eval/meridian_eval_set.jsonl")}
Q = lambda qid: EVAL[qid]["inputs"]["question"]
aug = AugmentedRAG(cfg)

def checks_table(checks):
    return pd.DataFrame([asdict(c) for c in checks])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: The problem
# MAGIC Answer Q38 with the Phase 8 pipeline and look at what each citation points to. Models vary from run to run, so it may or
# MAGIC may not cite the stale FAQ this time.

# COMMAND ----------

p38 = aug.plan(Q("Q38")); docs38 = aug.retrieve_plan(p38); msgs38 = aug._build_prompt(Q("Q38"), docs38, p38)
draft38 = aug.generate(msgs38)
print(draft38, "\n")
for n in cited_numbers(draft38):
    if 0 < n <= len(docs38):
        md = docs38[n - 1].metadata
        print(f"[{n}] {md['doc_uri']:24} status={md.get('status')}  authority_rank={md.get('authority_rank')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Fast checks, no model calls
# MAGIC Five answers to the same question, built by hand so every check fires at least once. The sources are the real ones
# MAGIC retrieved for Q16 (medium-risk review cycles).

# COMMAND ----------

p16 = aug.plan(Q("Q16")); docs16 = aug.retrieve_plan(p16)
for i, d in enumerate(docs16, start=1):
    print(f"[{i}] {d.metadata['doc_uri']:24} status={d.metadata.get('status')}  authority_rank={d.metadata.get('authority_rank')}")
faq_n = next((i for i, d in enumerate(docs16, start=1) if int(d.metadata.get("authority_rank") or 3) >= 4), None)
strong = [i for i, d in enumerate(docs16, start=1)
          if int(d.metadata.get("authority_rank") or 3) <= 2 and d.metadata.get("status", "current") == "current"]
s1 = strong[0] if strong else 1
s2 = strong[1] if len(strong) > 1 else s1
print(f"Binding current sources: {strong}; first FAQ/bulletin: {faq_n}")

CRAFTED = {
    "clean": f"Medium-risk clients are reviewed every 24 months [{s1}].",
    "cites the stale FAQ": f"Medium-risk clients are reviewed every 24 months [{s1}]. The FAQ says every three years [{faq_n or s1}].",
    "no citations": "Medium-risk clients are reviewed every 24 months.",
    "cites a missing source": f"Medium-risk clients are reviewed every 24 months [{s1}][{len(docs16) + 4}].",
    "messy format": f"Medium-risk clients are reviewed every 24 months [{s1}, {s2}].\n\nSources: [{s1}], [{s2}]",
}
rows = []
for label, ans in CRAFTED.items():
    cleaned, checks, notes = deterministic_checks(ans, docs16, p16)
    fired = [f"{c.name}: {c.action}" for c in checks if c.action != "none" or not c.passed]
    rows.append({"answer": label, "checks_fired": "; ".join(fired) or "none", "repair_needed": bool(notes), "after_cleanup": cleaned})
display(pd.DataFrame(rows))

# COMMAND ----------

# MAGIC %md
# MAGIC Some problems are fixed in place: bad citations are stripped and formatting is normalised. Others need the model to
# MAGIC rewrite, so they become **reviewer notes** for a single repair attempt: a weaker source cited while a binding one was
# MAGIC available, a known stale statement, or no citations at all.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Redact personal data
# MAGIC Applied to every answer, last, whatever else happened. Patterns catch identifiers with a fixed shape.

# COMMAND ----------

sample = ("The director, Lim Jia Hui, passport E7741203K, date of birth: 14 March 1979, can be reached at "
          "lim@halcyon.example or +65 9123 4567. The waiver limit is USD 20,000,000.")
redacted, found = redact_pii(sample)
print(redacted); print("Found:", found)

# COMMAND ----------

# MAGIC %md
# MAGIC Notice what isn't caught: the **name**. Names have no fixed shape, so patterns can't find them reliably. For that you'd
# MAGIC use entity recognition, such as the `ai_mask` SQL function. The amount stays, because a redactor that removes policy
# MAGIC numbers is as harmful as one that misses a passport number.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: The groundedness gate
# MAGIC One small model call checks whether every statement in the answer is supported by the sources. Here's a real-looking
# MAGIC answer with one invented requirement.

# COMMAND ----------

invented = (f"Medium-risk clients are reviewed every 24 months [{s1}]. Each review must also be signed off by the Chief "
            f"Executive Officer before the account can remain open [{s1}].")
for label, ans in [("supported", f"Medium-risk clients are reviewed every 24 months [{s1}]."), ("one invented claim", invented)]:
    g = check_groundedness(cfg, ans, docs16)
    print(f"{label:20} grounded={g.passed}  unsupported: {g.detail or '-'}")

# COMMAND ----------

# MAGIC %md
# MAGIC Citation markers are removed before the check, so the gate judges facts, not which number was cited. **Guardrails need
# MAGIC measuring too:** if the supported answer is flagged, that's a false positive, and in Step 6 it would cost a good answer.
# MAGIC The gate fails open: if the checker itself errors, the answer is released rather than blocked. In production you'd alert
# MAGIC on checker errors rather than silently pass them.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: The guarded pipeline, end to end
# MAGIC `guarded` runs every check, repairs at most once with reviewer notes, re-checks, and replaces the answer with a safe
# MAGIC fallback if it still isn't grounded. Open a trace: the `postprocess` span lists every check that fired.

# COMMAND ----------

guarded = GuardedRAG(cfg)
rows = []
for qid in ["Q38", "Q16", "Q09"]:
    p = guarded.plan(Q(qid)); docs = guarded.retrieve_plan(p); msgs = guarded.build_prompt(Q(qid), docs, p)
    draft = guarded.generate(msgs)
    result = guarded.post(draft, docs, p, msgs)
    rows.append({"id": qid, "outcome": result["outcome"], "check_seconds": result["seconds"],
                 "checks": "; ".join(f"{c['name']}: {c['action']}" for c in result["checks"]) or "all passed"})
    print(f"===== {qid} ({result['outcome']})\n--- draft:\n{draft}\n--- sent:\n{result['answer']}\n")
display(pd.DataFrame(rows))

# COMMAND ----------

# MAGIC %md
# MAGIC **Look at `check_seconds`.** Checks cost time: the groundedness gate is one extra model call, and a repair is another
# MAGIC full generation plus a second check. That's why repair is limited to one attempt.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Launch the after_p10 evaluation
# MAGIC The full pipeline: query plan, retrieval, augmented prompt, Llama 3.3 70B, and post-processing. From this run on, a new
# MAGIC `authoritative_citations` scorer also checks for weaker sources cited when a binding one was available. It runs while we
# MAGIC cover Phase 11.

# COMMAND ----------

launch_job("run_evaluation", {"pipeline": "guarded", "run_name": "after_p10"}, run_name="meridian-after-p10-eval")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC | You did | Why it matters |
# MAGIC |---|---|
# MAGIC | Checked citations and source authority | An instruction isn't enforcement; a check is |
# MAGIC | Flagged known stale and draft statements | The trap facts can't slip through unnoticed |
# MAGIC | Gated on groundedness | Unsupported claims get repaired or replaced |
# MAGIC | Repaired at most once | Bounded cost and latency |
# MAGIC | Redacted personal data last | Applied to every answer, whatever else happened |

# COMMAND ----------

import mlflow
for rn in ["baseline", "after_p2", "after_p3"]:
    r = mlflow.search_runs(experiment_names=[cfg.experiment_path],
                           filter_string=f"tags.mlflow.runName = '{rn}'",
                           order_by=["start_time DESC"], max_results=1).iloc[0]
    print(rn, {k.replace("metrics.", ""): round(v, 3) for k, v in r.items()
               if k.startswith("metrics.") and k.endswith("/mean") and "by_category" not in k})

for name, table in [("baseline", "naive_chunks"), ("after_p2", "p2_chunks_embedded"), ("after_p3", "p3_emb_structure")]:
    d = retrieval_eval_ranked(cfg, table)
    v = d.version_ok.dropna()
    print(name, f"right version first {int(v.sum())}/{len(v)}")