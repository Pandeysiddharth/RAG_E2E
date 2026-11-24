# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 11: Monitoring
# MAGIC
# MAGIC **Time box:** 10 minutes
# MAGIC
# MAGIC The copilot is "in production". There are no expected answers any more, only real questions. You'll send a batch of
# MAGIC simulated traffic through the guarded pipeline, score it with the scorers that **don't need ground truth**, record the
# MAGIC results in a monitoring table with alert thresholds, ship a deliberately bad change and watch the alerts fire, and turn a
# MAGIC bad production answer into a new evaluation record.
# MAGIC
# MAGIC **Done when:** you know what's running, you've seen an alert fire on the bad change, and you've added one production trace
# MAGIC to the evaluation set.

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
cfg, index = guard_lab11()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: What's running in production?
# MAGIC Before sending traffic, look at exactly what's being monitored. The `guarded` pipeline wasn't written in one go: each
# MAGIC lab added one layer, and Lab 10 registered the finished pipeline as `PIPELINES["guarded"]`.

# COMMAND ----------

rag = PIPELINES["guarded"](cfg)          # the same object Lab 10 built and evaluated as after_p10
display(pd.DataFrame(PIPELINE_LAYERS, columns=["layer", "added in", "what it adds"]))
display(describe_pipeline(cfg, rag))

# COMMAND ----------

# MAGIC %md
# MAGIC The path one request takes. Each call below is a traced method, so each becomes a span in the trace:

# COMMAND ----------

import inspect
print(inspect.getsource(GuardedRAG._answer_planned))

# COMMAND ----------

# MAGIC %md
# MAGIC ### The production scorers
# MAGIC Production traffic has no expected answers, so we keep only scorers that work without them. `monitor_scorers(cfg)`
# MAGIC builds this list from the evaluation scorers, plus a new judge. Built by hand, it would look like this:
# MAGIC
# MAGIC ```python
# MAGIC from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery, Safety
# MAGIC scorers = [RetrievalGroundedness(), RelevanceToQuery(), Safety(),      # built-in LLM judges
# MAGIC            cites_sources, authoritative_citations,                       # code checks (@scorer functions)
# MAGIC            refusal_appropriate]                                          # custom LLM judge
# MAGIC mlflow.genai.evaluate(data=production_traces, scorers=scorers)          # no predict_fn: score existing traces
# MAGIC ```

# COMMAND ----------

scorers = monitor_scorers(cfg)
display(pd.DataFrame([{"scorer": s.name, "kind": SCORER_GUIDE.get(s.name, ("?",))[0], "reads": SCORER_GUIDE.get(s.name, ("", "?"))[1],
                       "needs ground truth": SCORER_GUIDE.get(s.name, ("", "", "?"))[2]} for s in scorers]))
print("The refusal_appropriate judge's instructions:\n\n" + REFUSAL_JUDGE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### One request, one trace
# MAGIC Send a single question and look at the trace it produced. Every scorer above works by reading these spans.

# COMMAND ----------

import mlflow
answer = rag.answer("Is applicant consent needed for a USD 500k waiver?")
print(answer, "\n")
trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
for sp in trace.data.spans:
    print(f"{str(sp.span_type):12} {sp.name}")

# COMMAND ----------

# MAGIC %md
# MAGIC Open the trace in the experiment's **Traces** tab to see each span's inputs and outputs. Now send real traffic.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Production traffic
# MAGIC Fifteen questions like the ones real users send: paraphrases of known questions, new ones, out-of-scope ones, and a few
# MAGIC adversarial ones (asking for personal data, treating a draft as approved, asking the copilot to decide). They go through
# MAGIC the `guarded` pipeline, one trace each, tagged with a batch name. This takes about a minute.

# COMMAND ----------

display(pd.DataFrame(TRAFFIC, columns=["kind", "question"]))

# COMMAND ----------

week1, week1_ids = simulate_traffic(cfg, batch="week_1", pipeline="guarded")
display(week1.assign(answer=week1.answer.str[:140] + "…"))

# COMMAND ----------

# MAGIC %md
# MAGIC Look at the adversarial rows. The copilot should decline to list passport numbers, refuse to treat the 2026 draft as
# MAGIC approved, and explain the policy rather than approve the client itself.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Score without ground truth
# MAGIC In production nobody has written the right answer, so `Correctness` and `recall@5` can't run. These can: groundedness,
# MAGIC relevance, safety, citations, authority, and a new judge, **refusal_appropriate**, that asks whether answering or
# MAGIC declining was the right call. Operational metrics come from the traces: latency and repair rates.

# COMMAND ----------

week1_metrics, week1_eval = score_batch(cfg, "week_1")
display(pd.DataFrame([week1_metrics]).T.reset_index().rename(columns={"index": "metric", 0: "week_1"}))
print("Alerts:", alert_check(week1_metrics) or "none")

# COMMAND ----------

# MAGIC %md
# MAGIC Open the evaluation run linked above and read a few `refusal_appropriate` rationales, especially for the adversarial
# MAGIC questions. This judge replaces the keyword-based abstention check, which wrongly treated answers like "cannot be waived"
# MAGIC as refusals.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Ship a bad change
# MAGIC Someone "simplifies" the pipeline: same retrieval, but the naive prompt and no post-processing. That's our `retrieval`
# MAGIC pipeline from Phase 7. Same fifteen questions, new batch.

# COMMAND ----------

week2, week2_ids = simulate_traffic(cfg, batch="week_2_change", pipeline="retrieval")
week2_metrics, _ = score_batch(cfg, "week_2_change")
display(monitoring_view(cfg))
print("Alerts after the change:")
for a in alert_check(week2_metrics) or ["none"]:
    print("  ⚠️ ", a)

# COMMAND ----------

# MAGIC %md
# MAGIC **What to look for:** `cites_sources` should collapse, because the naive prompt never asks for citations, and
# MAGIC `refusal_appropriate` may drop on the adversarial questions. These are exactly the regressions an offline evaluation
# MAGIC would also catch, if someone ran it. Monitoring catches them even when nobody does.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: From a bad answer to a new test
# MAGIC Pick a production answer you'd mark thumbs-down. We record the feedback on its trace, then turn it into an evaluation
# MAGIC record so every future change is tested against it. That's the evaluation-first cycle: production failures become
# MAGIC tomorrow's eval set.

# COMMAND ----------

import mlflow
from mlflow.entities import AssessmentSource
# In the dry run, the guarded pipeline told a user that the Trade Operations Team Lead signs off a USD 3m waiver.
# Wrong: USD 3m is in the USD 1m-5m band, which the Country Head of Trade Operations approves. Read your answer first.
BAD_QUESTION = "Who signs off a USD 3m discrepancy waiver on a letter of credit?"   # or the "2026 draft" adversarial question
pick = int(week1.index[week1.question == BAD_QUESTION][0])
bad_trace = week1_ids[pick]
print("Question:", week1.question[pick]); print("Answer:", week1.answer[pick][:500])

# COMMAND ----------

# MAGIC %md
# MAGIC If the answer is wrong, record the thumbs-down. (If your copilot got it right this time, switch `BAD_QUESTION` to the
# MAGIC other question in `FEEDBACK_CASES`, or any answer you'd mark down.)

# COMMAND ----------

try:
    mlflow.log_feedback(trace_id=bad_trace, name="user_thumbs", value=False,
                        source=AssessmentSource(source_type="HUMAN", source_id="rm_demo"),
                        rationale="Wrong approver for the waiver amount band")
    ok("thumbs-down recorded on the trace")
except Exception as e:  # noqa: BLE001
    warn(f"couldn't log feedback ({str(e)[:80]})")

# COMMAND ----------

facts, reference, category = FEEDBACK_CASES[BAD_QUESTION]
record = trace_to_eval_record(bad_trace, expected_facts=facts, reference_answer=reference, category=category)
add_eval_records(cfg, [record])
print(json.dumps(record, indent=2)[:800])

# COMMAND ----------

# MAGIC %md
# MAGIC The guarded pipeline passed every check on this answer: the groundedness gate didn't notice the amount fell in the wrong
# MAGIC band. No set of checks is complete. Monitoring and user feedback find what the evaluation set never asked, and each
# MAGIC failure becomes a test that every future change has to pass.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Keep watching
# MAGIC In production you'd score traffic continuously or on a schedule. MLflow can register scorers to sample production
# MAGIC traces automatically; where that isn't available, a scheduled job running `score_batch` on each day's traces does the
# MAGIC same. Let's see whether registration works in this workspace.

# COMMAND ----------

try:
    from mlflow.genai.scorers import ScorerSamplingConfig
    s = [x for x in monitor_scorers(cfg) if x.name == "cites_sources"][0]
    registered = s.register(name="meridian_cites_sources")
    registered.start(sampling_config=ScorerSamplingConfig(sample_rate=0.2))
    ok("cites_sources registered to score 20% of new traces automatically")
except Exception as e:  # noqa: BLE001
    warn(f"Automatic scorer registration isn't available here ({str(e)[:100]}). Schedule score_batch as a daily job instead.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC | You did | Why it matters |
# MAGIC |---|---|
# MAGIC | Inspected the pipeline and its scorers | You can only monitor what you understand |
# MAGIC | Sent realistic and adversarial traffic | Production questions aren't the eval set |
# MAGIC | Scored without ground truth | The same scorers, minus the ones that need expected answers |
# MAGIC | Judged refusals with a model | Keyword heuristics misfire; a judge reads intent |
# MAGIC | Recorded metrics with thresholds | A drop becomes an alert, not a surprise |
# MAGIC | Shipped a bad change and caught it | Monitoring works even when nobody runs the evaluation |
# MAGIC | Turned a bad answer into a test | Production failures become tomorrow's eval set |
