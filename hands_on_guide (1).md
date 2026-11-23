# Meridian RAG Workshop: Hands-on Guide

This guide walks you through every lab, step by step. Each lab follows the same pattern: a guard cell that checks you're ready, a set of guided steps, and a final cell that launches the next long-running job in the background.

You'll work in your own Databricks Free Edition workspace on your own laptop, so nothing you do affects anyone else.

**Sections so far:** Before the session · Lab 0: Baseline · Lab 1: Corpus ingestion · Troubleshooting

---

## Before the session

**Time needed:** about 20 minutes, of which 10 are waiting. Please complete this at least a day before the workshop.

### 1. Create your Free Edition account

1. Go to the Databricks Free Edition sign-up page and sign up with a **personal** email address, or with Google or Microsoft sign-in. Don't use your corporate account.
2. When your workspace opens, look for **Verify identity** in the top bar. Click it and verify through LinkedIn. This raises some of your usage limits, which helps on the day.

### 2. Bring the workshop bundle into your workspace

Choose one option.

**Option A, Git folder (recommended).**
1. In the left menu, open **Workspace**, then your home folder.
2. Click **Create**, then **Git folder**.
3. Paste the workshop repository URL your instructor sent you and click **Create Git folder**.

**Option B, zip upload.**
1. Download `meridian_workshop_bundle.zip` and unzip it on your laptop.
2. In **Workspace**, open your home folder, click the three-dot menu and choose **Import**.
3. Import the whole `meridian-rag-workshop` folder, keeping its structure.

Either way, you should now see a folder containing `corpus`, `incoming`, `eval`, `checkpoints`, `glossary`, `reference` and `notebooks`.

### 3. Run the setup notebook

1. Open `notebooks/00_setup`.
2. At the top, make sure the compute selector shows **Serverless**.
3. Click **Run all**.

Every step prints a green tick when it succeeds. The final cell should print **All set. See you at the workshop.**

| Step | What it does | Expected time |
|---|---|---|
| 1 | Creates schema `workspace.meridian_rag` and volumes `raw` and `workshop` | Seconds |
| 2 | Copies 31 corpus files and the supporting files into the volumes | Under a minute |
| 3 | Finds and tests a chat model and an embedding model, then tunes the embedding batch size to your rate limit | About a minute |
| 4 | Starts creating your AI Search endpoint | Seconds (it finishes in the background) |
| 5 | Builds the naive baseline chunks (seconds if pre-computed embeddings are in the bundle; otherwise 1 to 3 minutes with progress shown) | Seconds to 3 minutes |
| 6 | Final check | Seconds |

> **Note:** the AI Search endpoint can take 10 to 20 minutes to come online after setup finishes. You don't need to wait for it. It isn't used until Phase 5.

### On the day

Bring your laptop charged, with your Free Edition workspace already signed in. If the venue Wi-Fi blocks Databricks, use your phone's hotspot.

---

## Lab 0: Baseline and evaluation set

**Notebook:** `01_baseline` · **Time box:** 12 minutes

**Goal:** measure a deliberately naive RAG pipeline before changing anything, so every later improvement can be proven.

### Step 1: Run the guard cell

Run the install cell, then the guard cell.

**Expected output:**
```
Guard check for Lab 0
  ✅ config loaded (chat: <your chat model>, embeddings: <your embedding model>)
  ✅ naive_chunks has 58 chunks
  ✅ MLflow experiment: /Users/<you>/meridian_rag_workshop
```

If the naive chunks are missing, the guard rebuilds them from checkpoint `cp_00` automatically.

### Step 2: Load the eval set into MLflow

Run the cells in Step 2. They load 40 questions from `meridian_eval_set.jsonl` and store them as a Unity Catalog evaluation dataset called `meridian_eval`.

Look at the first record. Each record has `inputs` (the question) and `expectations`:

| Field | What it's for |
|---|---|
| `expected_facts` | Atomic facts the answer must contain; used by the Correctness judge |
| `expected_doc_ids` | Documents that must be retrieved; used by the recall@5 scorer |
| `should_abstain` | Whether the copilot should answer, decline, or deflect a decision |
| `forbidden_content` | Strings that must never appear, such as numbers from an unapproved draft |

> **If the dataset can't be created**, you'll see a warning. Carry on: the evaluation job reads the JSONL file directly, so your results are the same.

### Step 3: Inspect the naive chunks

Run the three cells in Step 3 and answer these questions for yourself:

1. **Can you read the waiver matrix?** Each table cell has become its own line, so the link between an amount band and its approver is lost.
2. **Which files produced no chunks?** You should see two: the scanned UAE addendum and the photographed trade desk card. Both are images, so plain text extraction returns nothing.
3. **What's searchable that shouldn't be?** The 2026 consultation draft, the staff travel policy and a client case file with personal data.

### Step 4: Ask three questions and read the traces

Run the cell. For each answer, open the trace that appears below it and click the `retrieve` span.

| Question | Correct answer | What usually happens | What to look for in the trace |
|---|---|---|---|
| Q01: high-risk refresh cycle | 12 months | Right, but also mentions a 9-month cycle | A chunk from the unapproved 2026 draft |
| Q09: approver for a USD 4m waiver | Country Head of Trade Operations | Often right with a strong model | Whether the superseded 2024 matrix is retrieved too |
| Q23: cash-intensive UAE free-zone client | High risk by default; 6-monthly activity review | A generic EDD answer missing both rules | No UAE chunk at all: the scanned addendum produced no text |

> **Key insight:** when the answer is wrong, the model usually did its job with the evidence it was given. The trace shows the problem is retrieval, and the cause is upstream in the data.

### Step 4b: Check the scorers

Run the cell. It answers one question and runs all eight scorers on it, which takes about 10 seconds. Every line should be green, ending with **All scorers healthy.** If any line is red, tell your instructor before launching the job, because a scorer that fails here would silently drop out of the full evaluation.

### Step 5: Launch the baseline evaluation job

Run the cell. You'll see a link to the job run.

**Expected output:**
```
🚀 Launched 'meridian-baseline-eval' (run 123456789). It keeps running while we cover the next phase.
```

The job runs all 40 questions and scores them with five built-in LLM judges (Correctness, RetrievalSufficiency, RetrievalGroundedness, RelevanceToQuery, Safety) and three code scorers (recall_at_5, abstention_check, forbidden_content).

**Done when:** the evaluation dataset exists and the job shows as running.

### Optional: while you wait

Re-run the last cell after a few minutes. When the job finishes, your baseline metrics appear in the `scorecard` table. To browse per-question results, open **Experiments** in the left menu, then `meridian_rag_workshop`, then the **baseline** run.

---

## Lab 1: Corpus ingestion

**Notebook:** `02_ingestion` · **Time box:** 10 minutes

**Goal:** land every raw file in a Bronze Delta table with Auto Loader, prove the corpus covers the questions users ask, and record scope decisions.

### Step 1: Guard cell and status check

The guard tells you how your baseline evaluation is doing:

| Message | Meaning | What to do |
|---|---|---|
| ✅ Baseline evaluation finished | Your numbers are in the scorecard | Share them when the instructor asks |
| ⚠️ Still running | Normal if Lab 0 finished recently | Carry on; it doesn't block this lab |
| ❌ Ended with FAILED | Something went wrong | Raise a hand; carry on with the lab |

### Step 2: See what landed

Run both cells. You should see five file types (pdf, docx, html, pptx, png) across 31 files. Notice that one file, `Global_CDD_Policy_v2025_FINAL_copy.pdf`, doesn't follow the naming convention.

### Step 3: Ingest into Bronze with Auto Loader

Run the cell that defines and calls `ingest_bronze()`.

**Expected output:**
```
  ✅ Auto Loader processed 31 new file(s). workspace.meridian_rag.bronze_documents now has 31 rows.
```

The Bronze table keeps the raw bytes (`content`) along with `path`, `content_sha256`, `modificationTime`, `doc_id_hint` and `ingested_at`.

### Step 4: Check coverage against the eval set

Run the coverage cell.

**Expected output:**
```
The eval set needs 20 distinct documents; Bronze has 19 of them.

  ❌ MGB-KYC-ADD-UK is missing. Questions that can never be answered: Q25
```

> **Key insight:** no later phase can fix a missing document. This is why corpus composition starts from the questions.

### Step 5: Land the missing file and re-run

Run the cell that copies the UK addendum from `incoming` into the raw volume and calls `ingest_bronze()` again.

**Expected output:**
```
  ✅ Landed MGB-KYC-ADD-UK_UK_KYC_Addendum.html in the raw volume
  ✅ Auto Loader processed 1 new file(s). workspace.meridian_rag.bronze_documents now has 32 rows.
```

Exactly one file is processed, because the Auto Loader checkpoint remembers the other 31. Then run the next cell, which should confirm every document the eval set needs is now in Bronze.

### Step 6: Record scope decisions

Run the registry cell. It writes `source_registry`, marking each file as in or out of scope with a reason. Check that `MGB-HR-071` (the travel policy) is out of scope.

The draft and the case file stay in scope for now. Phase 2 filters by content: drafts by status, personal data by sensitivity, and near-duplicates by similarity.

**Try it:** in **Catalog**, open `bronze_documents` and select the **Lineage** tab.

### Step 7: Launch the parse job

Run the final cell. It launches `parse_documents`, which runs `ai_parse_document` over the in-scope files and writes `silver_parsed` and `silver_parsed_html` for Lab 2.

**Done when:** Bronze has 32 rows, `source_registry` exists, and the parse job is running.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No workshop config found` | Run `00_setup` first. |
| `ModuleNotFoundError: meridian_utils` | Open the notebook from inside the bundle's `notebooks` folder, not a copy elsewhere. |
| The guard says a table is missing | Re-run the guard with `use_checkpoint=True` (Lab 0) or set `RESET = True` (Lab 1). |
| Compute shuts down with a quota message | Free Edition has a daily usage quota. Pair with a neighbour for the rest of the session; your data is kept. |
| Model endpoint errors | Re-run step 3 of `00_setup` to re-detect available models. |
| Embedding shows `rate limited` retries | Normal on Free Edition: it backs off and lowers the batch size automatically. If it stops, wait a minute and re-run the cell. Embeddings are cached, so it resumes where it left off. |
| The baseline job failed | Open the run link, read the error, and ask a helper. It doesn't block later labs. |
| A metric is missing from the baseline run | A scorer errored on every question. Run Step 4b to see the full error. |
| A `CommandContext accessor` warning appears above your answers | Harmless: MLflow is looking for notebook details it doesn't need on serverless. Ignore it. |
| Auto Loader processed 0 files on the first run | The checkpoint already exists from an earlier attempt. That's fine if Bronze has rows; otherwise set `RESET = True`. |
