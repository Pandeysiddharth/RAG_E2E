# Building Production-Grade RAG on Databricks

**A reference guide for the Standard Chartered CIB workshop**

This guide is the reference companion to the workshop deck and the hands-on labs. The deck explains each phase in a few minutes and the labs let you practise it; this guide goes deeper, covering the reasoning behind each design choice, the trade-offs, and what changes when you move from the workshop to a production deployment. It follows the same chapter order as the workshop, and uses the same running example: a policy copilot for the onboarding and trade operations teams of Meridian Global Bank, a fictional bank whose 32 synthetic documents are described in the corpus specification.

**Chapters so far:** 0. Baseline and evaluation set · 1. Corpus ingestion

---

## Chapter 0: Baseline and evaluation set

### Why evaluation comes first

Most teams that already run RAG in production built it the same way. They assembled a pipeline, asked it a handful of questions, judged the answers by eye, and shipped when the answers looked right. Every later change, whether a new chunk size, a different embedding model or a rewritten prompt, was judged the same way. This is the vibe check, and its weakness is not that people judge badly. It is that the questions change from person to person and sprint to sprint, there is no record of what was asked, and a change that fixes one kind of question while quietly breaking another goes unnoticed.

Evaluation-driven development replaces opinion with a repeatable measurement. Before optimising anything, you fix a set of questions with known answers, run the current pipeline against them, and record the scores. That first run is the baseline. Every later change becomes an experiment with a before and after that anyone can inspect. This is the single biggest difference between a RAG prototype and a RAG system a bank can defend to its auditors, and it is why the workshop starts here rather than at the end.

### What a good evaluation set looks like

An evaluation set is only as useful as the questions in it. The Meridian set has 40 questions in seven categories, deliberately weighted towards hard cases: facts that live only inside tables, questions where the superseded and current versions of a policy disagree, rules that a jurisdiction addendum overrides, acronym-heavy phrasing, answers that need two documents, and questions the copilot should decline. A set made only of easy questions produces flattering scores and hides exactly the weaknesses you need to find.

Each record carries more than a question and an answer. The `expected_facts` field lists the atomic facts a correct answer must contain, which lets a judge award partial credit and makes its verdict far more reliable than comparing against one long reference answer. The `expected_doc_ids` field names the documents that must be retrieved, which allows retrieval to be scored separately from generation. It deliberately uses document identifiers rather than chunk identifiers, because chunk identifiers change every time chunking changes, and you want the retrieval score to stay comparable across every experiment. The `should_abstain` field says whether the copilot should answer, decline entirely, or explain the policy while leaving the decision to the designated approver. Finally, `forbidden_content` lists strings that must never appear, such as the 9-month review cycle from an unapproved draft, or a passport number from a client file. A partly correct answer that also contains forbidden content still fails, which catches errors that a correctness judge alone might miss.

In a real engagement, the questions come from the people who will use the copilot. A practical approach is to ask relationship managers and operations analysts for the twenty questions they ask most often, then add the hard cases yourself. Fifty to a hundred well-chosen questions are enough to drive development; more matter less than coverage of the failure modes you care about.

### How evaluation runs in MLflow 3

On Databricks, evaluation is a single call to `mlflow.genai.evaluate`. You pass it the evaluation data, a prediction function that calls your application once per question, and a list of scorers. MLflow traces every call, so each result is linked to the full record of what happened: the retrieved chunks, the prompt, the model response, token counts and latency.

Scorers come in two kinds. Built-in LLM judges assess qualities that need language understanding. In this workshop we use `Correctness` and `RetrievalSufficiency`, which compare against the expected facts, and `RetrievalGroundedness`, `RelevanceToQuery` and `Safety`, which need no ground truth. Code-based scorers are ordinary Python functions with the `@scorer` decorator, used for anything deterministic. We write three: `recall_at_5` reads the retriever span from the trace and checks whether the expected documents appear in the top five results; `abstention_check` checks whether the copilot declined when it should and answered when it should; and `forbidden_content` checks for strings that must never appear.

One detail matters more than it looks. The groundedness and sufficiency judges find the retrieved context by looking for a span of type `RETRIEVER` in the trace. If your retrieval function isn't traced with that span type, those judges have nothing to assess. The workshop's helper module marks the retrieval step correctly, and you should do the same in your own applications.

The same scorers are reused in Chapter 11 for production monitoring, where they run on a sample of live traffic. The effort you put into scorers during development therefore pays off twice.

### Reading a trace

A score tells you that something is wrong; a trace tells you where. The Phase 0 lab asks you to read the trace for a question the baseline gets wrong. The answer typically turns out to be fluent, confident and incorrect, but the generation step did nothing wrong: it faithfully summarised the evidence it was given. The retrieval span shows why. The superseded 2024 waiver matrix ranks first, the current 2025 matrix has been split into fragments, and the model had no version information to tell them apart. This pattern recurs throughout the workshop: most RAG quality problems originate upstream of the model, in the data and retrieval pipeline.

### The deliberately naive baseline

The baseline pipeline is intentionally naive, and roughly matches what many teams ship as a first version. It extracts plain text from each file, cuts the text into fixed 1,000-character pieces with no overlap, embeds them with one model, retrieves the top five by vector similarity with no filters, and asks the model to answer using the context with no further instructions. It does no filtering of drafts, duplicates or personal data, and no post-processing. Each of these choices is a knob that a later phase turns, and the scorecard records the effect of each one.

The baseline retrieves from an in-memory index over pre-computed embeddings, which makes it ready in seconds. The managed AI Search index is built in Chapter 5, once the chunks are worth indexing.

### Going further

In production, store the evaluation set as a Unity Catalog evaluation dataset so it is versioned and governed alongside the data it tests. Add questions whenever a user reports a bad answer, so the set grows to cover real failure modes. Run evaluation automatically whenever the pipeline changes, and treat a drop in any category as a failed build. Finally, have a domain expert review a sample of judge verdicts periodically; built-in judges are strong but not infallible, and calibrating them against expert opinion builds trust in the numbers.

---

## Chapter 1: Corpus ingestion

### The principle

A RAG application can only answer from what it has ingested. This sounds obvious, but the most damaging corpus problem is the document that isn't there, because nothing looks wrong. The pipeline runs, retrieval returns plausible chunks, and the model writes a confident answer from the wrong source. Corpus ingestion is therefore not just a loading step. It is where you decide what the copilot knows, and where you make sure it knows enough.

### Composition starts from the questions

Databricks' own RAG guidance recommends choosing sources based on the specific use case and involving domain experts early to understand what users will ask. In practice this means working backwards: start from the questions, identify which documents answer them, and only then build the pipeline.

The evaluation set from Chapter 0 makes this concrete. Because every question records the documents it depends on, it doubles as a coverage test: compare the documents the questions need against the documents you have ingested, and any gap appears immediately. In the workshop, this check reveals that the UK KYC addendum was never ingested, so question Q25 could never be answered no matter how well everything else was tuned.

Coverage is necessary but not sufficient. A good corpus also excludes what doesn't belong. The Meridian landing zone contains a staff travel policy, which is a fine document for a different assistant but invites off-topic answers in this one; an unapproved consultation draft; a client case file with personal data; a near-duplicate copy of the main policy; and a superseded version that is still needed for historical questions. Each requires a different decision: exclude, filter by status, filter by sensitivity, deduplicate, or keep with a status tag. The workshop records these decisions in a `source_registry` table rather than deleting files, so the choices are visible, auditable and reversible. Domain scope is decided in Chapter 1; content-based filtering of drafts, personal data and duplicates follows in Chapter 2, once the documents have been parsed.

### Landing raw files in Unity Catalog

Raw files land in a Unity Catalog volume, exactly as received. Volumes give files the same governance as tables: access control, auditing and lineage. In production, files typically arrive through a managed connector, such as Lakeflow Connect for SharePoint, or through an upstream process that writes to cloud storage. In the workshop, the setup notebook copies them from the bundle.

### Incremental ingestion with Auto Loader

Databricks recommends ingesting data incrementally and storing raw source data in a target table, which preserves the original, supports traceability and simplifies auditing. Auto Loader implements this pattern. Configured with the `cloudFiles` source and the `binaryFile` format, it reads each new file as bytes together with its path, size and modification time. It records every file it has processed in a checkpoint, so each file is processed exactly once.

Running Auto Loader with `trigger(availableNow=True)` processes everything new and then stops. This gives you the bookkeeping of a stream with the cost profile of a batch job, and it means the same code serves the initial load and every later top-up. In the lab, you run it once to load 31 files, then land the missing UK addendum and run the identical code again; the checkpoint ensures that only the one new file is processed.

### What goes into Bronze

The Bronze table keeps the raw bytes of every file, not just extracted text. This is the most important design decision in the chapter. Parsers improve constantly, and in Chapter 2 you will re-parse every document with `ai_parse_document`, a much stronger parser than the plain text extraction used by the baseline. Because the bytes are already in Bronze, that requires no change to the source systems and no re-ingestion.

Alongside the bytes, the workshop's Bronze table stores the file path, so every answer can be traced to the exact file; a SHA-256 hash of the content, which detects exact duplicates regardless of file name and shows when a file with the same name has actually changed; the modification time; a document identifier hint taken from the file name; and the ingestion timestamp, which answers the audit question of what the copilot knew and since when. The identifier hint is only a hint, because some files, such as the emailed "FINAL copy" of the policy, don't follow the naming convention. Chapter 2 extracts the authoritative identifier from the document itself.

Every table derived from Bronze extends the lineage that Unity Catalog records automatically, from the volume through Bronze to the registry, and on through the parsed, chunked and indexed tables to come. For a bank, this lineage is the basis for answering a regulator's question about where a given answer came from.

### Going further

In production, schedule ingestion as a job or a Lakeflow pipeline so new and updated documents flow through automatically. Use the content hash to trigger re-parsing and re-embedding only for files that have actually changed, which keeps costs proportional to change rather than to corpus size. Treat the source registry as a governed artefact owned jointly by the business and the platform team, with a review whenever a new source is proposed. And re-run the coverage check whenever the evaluation set grows, so new questions immediately expose missing documents.
