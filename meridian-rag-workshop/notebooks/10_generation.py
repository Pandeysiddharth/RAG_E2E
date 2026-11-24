# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Lab 9: LLM generation
# MAGIC
# MAGIC **Time box:** 7 minutes
# MAGIC
# MAGIC The prompt and retrieval from Phases 7 and 8 stay exactly as they are; only the **generation model** changes. You'll
# MAGIC compare models on the same evidence for quality, latency and tokens, see where the tokens in a request go, watch what
# MAGIC temperature does, and launch full evaluations of the alternatives.
# MAGIC
# MAGIC **Done when:** you've compared models side by side and the model-comparison job is running.

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Guard cell, and which models do we have?

# COMMAND ----------

import importlib, meridian_utils
importlib.reload(meridian_utils)        # always use the file on disk
from meridian_utils import *
self_check()                            # clear message if meridian_utils.py is out of date
import pandas as pd
cfg, index = guard_lab9()
EVAL = {json.loads(l)["tags"]["question_id"]: json.loads(l) for l in open(f"{cfg.work_path}/eval/meridian_eval_set.jsonl")}

available = list_chat_models()
print("Chat models in this workspace:", available)
CANDIDATES = pick_candidates(cfg, available, n=2)      # edit this list to compare different models
MODELS = [cfg.chat_endpoint] + CANDIDATES
print("Comparing:", MODELS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Same evidence, different models
# MAGIC Each model gets **exactly** the same prompt: same query plan, same retrieved sources, same instructions. Four questions,
# MAGIC chosen for things a smaller model might drop: a condition (Q09, Q28), the approver deferral (Q38), and a multi-hop answer
# MAGIC needing two sources (Q31).

# COMMAND ----------

probe = generation_probe(cfg, MODELS, ["Q09", "Q28", "Q38", "Q31"], EVAL)
display(probe.drop(columns=["answer"]))

# COMMAND ----------

summary = probe.groupby("model").agg(median_seconds=("seconds", "median"), mean_completion_tokens=("completion_tokens", "mean"),
                                     cites=("cites", "mean"), truncated=("maybe_truncated", "sum")).round(2).reset_index()
display(summary)

# COMMAND ----------

for qid in ["Q09", "Q38"]:
    print(f"===== {qid}: {EVAL[qid]['inputs']['question']}")
    for _, r in probe[probe.id == qid].iterrows():
        print(f"--- {r.model} ({r.seconds}s, {r.completion_tokens} tokens):\n{r.answer}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC **Watch `maybe_truncated` and empty answers.** Reasoning models (such as GPT OSS) spend tokens thinking before they answer;
# MAGIC with too small a `max_tokens` they return nothing. The pipeline allows 1,000 output tokens (`GEN_MAX_TOKENS`).
# MAGIC
# MAGIC **Check each model:** does it keep the applicant's consent in Q09, the clock pause in Q28, the approver deferral in Q38,
# MAGIC both the approver and the credit concurrence in Q31, and `[n]` citations throughout? A faster model that drops a
# MAGIC condition is cheaper, but it's also wrong.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Where do the tokens go?
# MAGIC Cost and latency follow tokens. Here's one request broken down.

# COMMAND ----------

rag = AugmentedRAG(cfg)
q = EVAL["Q09"]["inputs"]["question"]
p = rag.plan(q); docs = rag.retrieve_plan(p); msgs = rag._build_prompt(q, docs, p)
context, ctx_tokens, n_src = build_context(docs)
answer_tokens = probe[(probe.id == "Q09") & (probe.model == short_model(cfg.chat_endpoint))].completion_tokens.iloc[0]
display(pd.DataFrame([
    {"part": "system prompt (instructions)", "approx_tokens": approx_tokens(msgs[0]["content"])},
    {"part": f"sources ({n_src} chunks)", "approx_tokens": ctx_tokens},
    {"part": "question and notes", "approx_tokens": approx_tokens(msgs[1]["content"]) - ctx_tokens},
    {"part": "answer (generated)", "approx_tokens": answer_tokens},
]))

# COMMAND ----------

# MAGIC %md
# MAGIC The sources dominate the input; the answer is a small share of the total. That's why the context budget from Phase 8,
# MAGIC and choosing k carefully in Phase 7, matter for cost as much as for quality.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Temperature
# MAGIC The same question, three times at temperature 0 and three times at 0.7, with the current model.

# COMMAND ----------

runs = []
for temp in (0.0, 0.7):
    for i in range(3):
        r = generation_probe(cfg, [cfg.chat_endpoint], ["Q38"], EVAL, temperature=temp)
        runs.append({"temperature": temp, "run": i + 1, "words": len(r.answer.iloc[0].split()),
                     "defers_to_approver": r.defers_to_approver.iloc[0], "answer": r.answer.iloc[0]})
runs = pd.DataFrame(runs)
display(runs.assign(answer=runs.answer.str[:120] + "…"))
print({t: runs[runs.temperature == t].answer.nunique() for t in (0.0, 0.7)}, "distinct answers (out of 3) per temperature")

# COMMAND ----------

# MAGIC %md
# MAGIC At temperature 0 the answers should be nearly identical, though not always exactly: hosted models aren't perfectly
# MAGIC deterministic, which is part of the run-to-run noise we've seen since Phase 2. At 0.7 they vary more. For policy answers,
# MAGIC keep temperature at 0.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Launch full evaluations of the alternatives
# MAGIC A four-question probe can't pick a winner. The job evaluates each candidate model on all 40 questions, one after another,
# MAGIC as runs named `after_p9_<model>`, with the same prompt, retrieval and scope check. Llama 3.3 70B's result is `after_p8`.

# COMMAND ----------

if CANDIDATES:
    launch_job("compare_models", {"chat_endpoints": ",".join(CANDIDATES)}, run_name="meridian-compare-models")
else:
    warn("No alternative chat models available in this workspace. Phase 9 still showed latency, tokens and temperature.")

# COMMAND ----------

# MAGIC %md
# MAGIC When the job finishes (a few minutes per model), compare everything in one table:

# COMMAND ----------

run_names = ["after_p8"] + [f"after_p9_{short_model(m).replace('-', '_')}" for m in CANDIDATES]
display(generation_report(cfg, run_names))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC | You did | Why it matters |
# MAGIC |---|---|
# MAGIC | Changed only the model | The comparison measures generation, not retrieval or prompt |
# MAGIC | Compared latency, tokens and completeness | The cheapest model is only cheap if it's still right |
# MAGIC | Saw where tokens go | Sources dominate the input, so context size drives cost |
# MAGIC | Compared temperatures | Keep temperature at 0 for policy answers |
# MAGIC | Evaluated alternatives on all 40 questions | A handful of examples can't pick a winner |

# COMMAND ----------

launch_job("compare_models", {"chat_endpoints": "databricks-gpt-oss-20b"}, run_name="meridian-gpt-oss-rerun")