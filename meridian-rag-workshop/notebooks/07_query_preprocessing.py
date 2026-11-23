# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Lab 6: Query preprocessing
# MAGIC
# MAGIC **Time box:** 10 minutes
# MAGIC
# MAGIC So far we've searched with the question exactly as typed. You'll build a **query plan** for each question before
# MAGIC searching: expand acronyms, work out which jurisdiction it's about, notice when it asks about the past, and route
# MAGIC questions the copilot shouldn't answer to a polite refusal. Then you'll measure what the plan does for retrieval and
# MAGIC launch a full evaluation.
# MAGIC
# MAGIC **Done when:** you've seen plans for in-scope and out-of-scope questions, and the `after_p6` evaluation is running.

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
from dataclasses import asdict
cfg, index = guard_lab6()
EVAL = {json.loads(l)["tags"]["question_id"]: json.loads(l) for l in open(f"{cfg.work_path}/eval/meridian_eval_set.jsonl")}
Q = lambda qid: EVAL[qid]["inputs"]["question"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Expand acronyms
# MAGIC Meridian's staff write in acronyms; its policies spell things out. In the kickoff's map of meaning, "Do we need SoW?"
# MAGIC landed between topics. We expand acronyms from the glossary before searching. It's deterministic: no model call.

# COMMAND ----------

glossary = load_glossary(cfg)
display(pd.DataFrame([{"question": Q(q), "rewritten": expand_acronyms(Q(q), glossary)[0]} for q in ["Q26", "Q27", "Q28", "Q29", "Q30"]]))

# COMMAND ----------

# MAGIC %md
# MAGIC Now compare what the index returns for Q30 as typed and as rewritten.

# COMMAND ----------

for label, text in [("as typed", Q("Q30")), ("rewritten", expand_acronyms(Q("Q30"), glossary)[0])]:
    print(f"--- {label}: {text}")
    for h in index_search(index, text, k=5, columns=["doc_key", "status"]):
        print(f"   {h['score']:.3f}  {h['doc_key']}")

# COMMAND ----------

# MAGIC %md
# MAGIC The expected source for Q30 is the TBML red flags guide, `MGB-TF-GDE-034`. Check where it ranks before and after.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Extract filters from the question
# MAGIC Two signals, found with simple rules:
# MAGIC - **Jurisdiction:** a question about Singapore should search Singapore and global documents only.
# MAGIC - **Historical intent:** "before", "previous", "what changed", or a year before 2025. Phase 7 will filter out superseded
# MAGIC   documents **by default**, so questions like Q15 must be flagged, or the 2024 policy they need would be filtered away.

# COMMAND ----------

rows = []
for qid in ["Q20", "Q21", "Q23", "Q25", "Q15", "Q18", "Q16", "Q01", "Q19", "Q09"]:
    f, jur, hist = extract_filters(Q(qid))
    rows.append({"id": qid, "question": Q(qid), "jurisdiction": jur or "-", "filters": f or "-", "historical": hist})
display(pd.DataFrame(rows))

# COMMAND ----------

# MAGIC %md
# MAGIC **Check two things:** Q15 ("before the 2025 policy update") is `historical`, and Q19 ("is a 9-month cycle **now** in
# MAGIC effect?") is not. Q16 has no jurisdiction, so it searches every jurisdiction: correct, but its answer should mention the
# MAGIC Hong Kong exception, which is a job for the prompt in Phase 8.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Route out-of-scope questions
# MAGIC One small model call decides whether the question belongs to this copilot. Out-of-scope questions get a polite,
# MAGIC specific refusal **without** searching, so the model can't improvise an answer from loosely related chunks. We check both
# MAGIC directions: the abstain questions should be refused, and ordinary policy questions must not be.

# COMMAND ----------

check = ["Q36", "Q37", "Q39", "Q40", "Q38", "Q01", "Q09", "Q20", "Q26", "Q34"]
plans = {qid: plan_query(cfg, Q(qid)) for qid in check}
display(pd.DataFrame([{"id": qid, "question": Q(qid), "expected": EVAL[qid]["expectations"]["should_abstain"],
                       "in_scope": p.in_scope, "category": p.scope_category, "reason": p.reason} for qid, p in plans.items()]))

# COMMAND ----------

# MAGIC %md
# MAGIC **What to look for:** Q36 (travel allowance), Q37 (passport number), Q39 (SOFR rate) and Q40 (quote an external rulebook)
# MAGIC should be out of scope, each with its own category. Q38 ("should I approve…") stays **in** scope: the copilot should explain
# MAGIC the policy and leave the decision to the approver, which Phase 8's prompt handles. Any policy question marked out of scope
# MAGIC is a false refusal, and worse than a missed abstention.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: What does the plan do for retrieval?
# MAGIC Retrieval-only evaluation through the index, with and without the plan's rewriting and filters. (The scope check isn't
# MAGIC needed here, so this makes no model calls beyond the index's own embeddings.)

# COMMAND ----------

no_plan = index_retrieval_eval(cfg, index)
with_plan = index_retrieval_eval(cfg, index, planner=lambda q: plan_query(cfg, q, use_llm=False))
display(summarise_models({"as typed": no_plan, "with query plan": with_plan}))

# COMMAND ----------

by_cat = pd.DataFrame({name: d.groupby("category")["reciprocal_rank"].mean().round(2)
                       for name, d in {"as typed": no_plan, "with query plan": with_plan}.items()})
display(by_cat.reset_index().rename(columns={"index": "category"}))

# COMMAND ----------

# MAGIC %md
# MAGIC Recall@5 was already at its ceiling, so look at **MRR** and **recall@1**, and at the jurisdiction and acronym categories,
# MAGIC where the plan should help most. Remember the noise rule: a difference of a question or two isn't evidence.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Try the preprocessed pipeline
# MAGIC The `preprocessed` pipeline plans the query, refuses out-of-scope questions, and otherwise searches with the rewritten
# MAGIC question and filters. Open each trace: the `plan_query` span shows the plan.

# COMMAND ----------

rag = PreprocessedRAG(cfg)
for qid in ["Q37", "Q36", "Q20", "Q30"]:
    print(f"--- {qid}: {Q(qid)}")
    print(rag.answer(Q(qid)), "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Launch the after_p6 evaluation
# MAGIC Same prompt and model as `after_p5`; the only change is the query plan. It runs while we cover Phase 7.

# COMMAND ----------

launch_job("run_evaluation", {"pipeline": "preprocessed", "run_name": "after_p6"}, run_name="meridian-after-p6-eval")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC | You did | Why it matters |
# MAGIC |---|---|
# MAGIC | Expanded acronyms from a glossary | The question's words now match the policies' words |
# MAGIC | Extracted jurisdiction filters | Searches only the documents that apply |
# MAGIC | Flagged historical questions | Lets Phase 7 filter superseded versions by default without breaking Q15 |
# MAGIC | Routed out-of-scope questions | A specific refusal, without retrieval, beats an improvised answer |
# MAGIC | Checked for false refusals | Refusing a legitimate question is worse than a missed abstention |