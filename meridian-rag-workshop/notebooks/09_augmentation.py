# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 8: Prompt augmentation
# MAGIC
# MAGIC **Time box:** 7 minutes
# MAGIC
# MAGIC Retrieval now puts the right evidence, in the right order, in front of the model. But the prompt is still the naive
# MAGIC "answer using the context". You'll compare it with an augmented prompt that numbers and labels each source, asks for
# MAGIC every applicable condition, states the precedence rules, requires citations, and says when not to answer. Same retrieval,
# MAGIC same model: only the prompt changes.
# MAGIC
# MAGIC **Done when:** you've compared both prompts on four questions and the `after_p8` evaluation is running.

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
cfg, index = guard_lab8()
EVAL = {json.loads(l)["tags"]["question_id"]: json.loads(l) for l in open(f"{cfg.work_path}/eval/meridian_eval_set.jsonl")}
Q = lambda qid: EVAL[qid]["inputs"]["question"]
naive_rag, aug_rag = RetrievalRAG(cfg), AugmentedRAG(cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Two prompts, same evidence
# MAGIC Q09 asks who can approve a USD 4 million waiver. In Lab 7 the copilot named the right approver but left out that the
# MAGIC applicant's written consent is required, a condition that was in the retrieved table. Here are both prompts, built from
# MAGIC exactly the same retrieved chunks.

# COMMAND ----------

plan = naive_rag.plan(Q("Q09"))
docs = naive_rag.retrieve_plan(plan)
naive_msgs = naive_rag._build_prompt(Q("Q09"), docs)
aug_msgs = aug_rag._build_prompt(Q("Q09"), docs, plan)
for label, msgs in [("NAIVE", naive_msgs), ("AUGMENTED", aug_msgs)]:
    tokens = sum(approx_tokens(m["content"]) for m in msgs)
    print(f"===== {label} prompt: about {tokens} tokens =====")
    print("[system]", msgs[0]["content"][:1200])
    print("[user]", msgs[1]["content"][:900], "...\n")

# COMMAND ----------

# MAGIC %md
# MAGIC **What changed:** every source is numbered and labelled with its document, version, status, jurisdiction and authority,
# MAGIC best first. The system prompt asks for conditions and exceptions, states the precedence rules, requires `[n]` citations,
# MAGIC and says what to do when the sources don't answer or a decision is being asked for.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Compare the answers
# MAGIC Four questions, each chosen for a failure we've seen in the dry runs.

# COMMAND ----------

CASES = {"Q09": "left out the applicant's consent",
         "Q28": "left out that the clock pauses",
         "Q38": "should explain the policy and leave the decision to the approver",
         "Q16": "should mention the Hong Kong exception (18 months)"}
rows = []
for qid, why in CASES.items():
    rows.append({"id": qid, "question": Q(qid), "watch_for": why,
                 "naive_prompt": naive_rag.answer(Q(qid)), "augmented_prompt": aug_rag.answer(Q(qid))})
answers = pd.DataFrame(rows)
for r in rows:
    print(f"===== {r['id']}: {r['question']}\n   watch for: {r['watch_for']}")
    print(f"--- naive:\n{r['naive_prompt']}\n--- augmented:\n{r['augmented_prompt']}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC **Check each pair:** does the augmented answer include the consent (Q09) and the pause (Q28)? Does it leave the
# MAGIC decision to the approver (Q38)? Does it mention Hong Kong's 18 months alongside the global 24 (Q16)? Also notice length:
# MAGIC asking for completeness can make answers longer. Phase 11's relevance judge will tell us if it's gone too far.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Do the citations point at real sources?
# MAGIC A citation is only useful if it refers to a source that was actually retrieved. We map each `[n]` in the augmented
# MAGIC answers back to its document.

# COMMAND ----------

for r in rows:
    p = aug_rag.plan(Q(r["id"]))
    if not p["in_scope"]:
        print(f"{r['id']}: refused without retrieval"); continue
    srcs = [d.metadata["doc_uri"] for d in aug_rag.retrieve_plan(p)]
    cited = cited_numbers(r["augmented_prompt"])
    mapped = {c: (srcs[c - 1] if 0 < c <= len(srcs) else "NOT A RETRIEVED SOURCE") for c in cited}
    print(f"{r['id']}: cites {mapped or 'nothing'}")

# COMMAND ----------

# MAGIC %md
# MAGIC From now on, every evaluation also runs a `cites_sources` scorer: it checks the answer cites at least one retrieved
# MAGIC source, and only sources that exist.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Version the prompt
# MAGIC A prompt is code: it changes behaviour as much as any function. We register it in the MLflow Prompt Registry in Unity
# MAGIC Catalog, and every evaluation run logs the prompt version it used.

# COMMAND ----------

prompt_ref = register_copilot_prompt(cfg)
print("Prompt reference:", prompt_ref)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Launch the after_p8 evaluation
# MAGIC Same query plan, same retrieval, same model as `after_p7`. Only the prompt is different. It runs while we cover Phase 9.

# COMMAND ----------

launch_job("run_evaluation", {"pipeline": "augmented", "run_name": "after_p8"}, run_name="meridian-after-p8-eval")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC | You did | Why it matters |
# MAGIC |---|---|
# MAGIC | Numbered and labelled every source | The model can see version, status, jurisdiction and authority, and cite them |
# MAGIC | Asked for every applicable condition | Answers stop dropping the consent or the exception |
# MAGIC | Stated the precedence rules | The model knows which source wins when they disagree |
# MAGIC | Said when not to answer | "The sources don't cover this" and "the approver decides" are correct answers |
# MAGIC | Versioned the prompt | Every result can be traced to the exact prompt that produced it |
