# VRHS Chatbot

A retrieval-augmented chatbot for Vista Ridge High School. It answers student and
parent questions from the school's own web pages, streams the answer token by
token, and then checks its own output for hallucination before asking anyone to
trust it.

Built by Saif Syed and Junayd Elhassan. The running deployment is linked in the
repository sidebar.

![The chatbot answering a question about attendance hours](docs/screenshot.png)

---

## Contents

- [How it works](#how-it-works)
- [The hallucination detection layer](#the-hallucination-detection-layer)
- [Choosing a detection architecture](#choosing-a-detection-architecture)
- [Decisions and what they measurably bought](#decisions-and-what-they-measurably-bought)
- [Where measurement overruled the design](#where-measurement-overruled-the-design)
- [Performance](#performance)
- [What is not measured](#what-is-not-measured)
- [Reproducing the numbers](#reproducing-the-numbers)

---

## How it works

Three stages: an offline ingest that builds the index, a per-question answer
path, and a verification layer that runs once the answer is complete.

![Architecture: ingest, answer path, and the hallucination layer](docs/architecture.svg)

The index is a JSON file of 44 chunks held in process. At this size a vector
database would add operational weight and no measurable speed: brute-force
cosine over 44 vectors of 1,536 dimensions takes under a millisecond, which is
three orders of magnitude below the network round trip that precedes it. The
constraint here is API latency, not search.

## The hallucination detection layer

The failure mode of a small RAG system is not wild invention. It is a fluent,
confident answer built on retrieved text that does not actually support it. For
a school chatbot the concrete version is an invented URL: a student follows
`vrhs.leanderisd.org/lunch-menu`, hits a 404, and stops trusting the tool.

Three checks run on every answer, cheapest first:

| Check | Question it answers | Cost | Measured |
| --- | --- | --- | --- |
| **Retrieval gate** | Does the corpus cover this at all? | free | 96% separation of answerable vs out-of-scope |
| **Link grounding** | Does every URL appear verbatim in the context? | free | precision 1.00, recall 1.00 |
| **Claim grounding** | Does the context support each claim? | $0.00007 | precision 1.00, recall 1.00 |

Two design choices worth stating:

**Checks annotate, they never retract.** Verification can only run once the
answer exists as a whole, and by then the tokens are already on the reader's
screen. Deleting text someone just read is worse than flagging it, so a failed
check appends a notice.

**The grader is not the generator.** Claim grounding runs on `gpt-4o-mini` while
answers come from `gpt-4o`. A grader sharing the generator's failure modes will
ratify its mistakes.

Here is the layer firing on a question the corpus genuinely does not cover:

![The verification notice appended to an ungrounded answer](docs/verification.png)

## Choosing a detection architecture

The LLM judge was not the obvious choice — it is the slowest and most expensive
of the options. So all four were built and measured against 20 labeled cases (9
answers with injected false claims, 11 grounded), drawn from real chunks in the
corpus.

| Architecture | Precision | Recall | F1 | p50 latency | $/check |
| --- | --- | --- | --- | --- | --- |
| Lexical overlap | 0.889 | 0.889 | 0.889 | 0 ms | $0 |
| Embedding similarity | 0.800 | 0.889 | 0.842 | 269 ms | $0.000014 |
| Cascade (lexical → LLM) | 1.000 | 0.889 | 0.941 | 0 ms | $0.000025 |
| **LLM judge (`gpt-4o-mini`)** | **1.000** | **1.000** | **1.000** | 541 ms | $0.000072 |

Reading this honestly:

- **Lexical overlap is a strong baseline.** F1 0.889 for zero cost and zero
  latency. Any claim that a model call is *necessary* has to beat this first.
- **Embedding similarity was the worst of the four.** Sentence vectors are close
  to the context whenever the topic matches, and an injected false claim usually
  keeps the topic. It fails exactly where it is needed.
- **The cascade is the interesting one.** A lexical prefilter resolves the
  confident cases and escalates only the ambiguous band to the model — 13 of 20
  cases never made an API call, giving F1 0.941 at a third of the cost and a 0 ms
  median.
- **The LLM judge ships** because at this volume the cost difference is
  meaningless. A school chatbot serving even 10,000 questions a month spends
  $0.72 on the judge versus $0.25 on the cascade. Paying $0.47 to recover the
  last 6 points of F1 is obviously right. **The cascade becomes the correct
  choice at roughly 100× this traffic**, and it is kept in `eval/detectors.py`
  for that reason.

The two threshold-based detectors are reported at their *best achievable* F1,
with the threshold chosen on this same 20-case set. That flatters them, and the
LLM judge still wins.

## Decisions and what they measurably bought

| Decision | Measured effect |
| --- | --- |
| Repaired 2 stale source URLs that 404'd | top-3 retrieval **53.3% → 66.7%** |
| Extended link checking to bare URLs | recall **0.818 → 1.000** |
| Absolute cosine gate instead of a standard-score margin | separation **84% → 96%** |
| Corpus hygiene pass before embedding | background cosine **0.892 → 0.788**, index **108 → 44 chunks**; top-3 −6.6 pts |
| LLM judge over the cheaper detectors | F1 **0.889 → 1.000** for +$0.00007 and +541 ms after the stream |
| Verify after the last token, not before the first | perceived wait unchanged at **757 ms** |
| Removed duplicate embedding of manual chunks | 5 fewer API calls per rebuild |

## Where measurement overruled the design

### Retrieval thresholds: wrong twice before being measured

**First attempt — intuition.** Cosine cutoffs of 0.82 and 0.76, picked from a
general sense of where ada-002 sits. No evidence.

**Second attempt — corpus analysis said absolute cutoffs were impossible.** On
the raw corpus the median cosine between two *unrelated* chunks was 0.8920,
while the 5th percentile of a *perfect* match was 0.8802. The distributions
overlap, so no single cutoff separates them. That argued for a scale-free
standard score, `(top − mean) / std`.

**Third attempt — real questions reversed it again.** Step 2 used chunks as
stand-in queries, and chunks are the one thing carrying the site's navigation
boilerplate that inflated the baseline. A real question is short and carries
none of it. Head to head over 25 questions:

| Signal | Best cutoff | Separation |
| --- | --- | --- |
| **Raw top-1 cosine** | 0.780 | **96%** |
| Standard-score margin | 2.42 | 84% |

The threshold that had been discarded was the right answer; the proxy experiment
was measuring the wrong thing. The lesson, recorded in `hallucination.py`: a
proxy measurement can be worse than no measurement, because it is persuasive.

### Corpus hygiene: three quarters of the corpus was navigation

A naive full-page scrape embeds the site's nav menu into every chunk. Both
corpora below come from the same scrape, so the comparison is clean:

| | Baseline | After hygiene |
| --- | --- | --- |
| Chunks | 108 | **44** |
| Words | 14,931 | **3,618** (75.8% removed) |
| Duplicate chunks | 17 | **0** |
| Words inside cross-page boilerplate | 75.9% | **0%** |
| Median cosine, unrelated pairs | 0.8920 | **0.7884** |
| Separable by an absolute cutoff? | no | **yes** |
| Top-1 retrieval | 53.3% | 53.3% |
| Top-3 retrieval | **73.3%** | 66.7% |

The pass does what it was built for: it collapses the similarity floor, which is
what makes the retrieval gate viable at all. But it also costs one question of
top-3 recall. At n=15 that is one item, well inside noise, and it is reported
rather than buried — the honest summary is that hygiene is a corpus-quality and
cost win that has *not* been shown to improve recall.

### Ingest coverage: two pages had been silently empty

The scraper wrapped each page in `try/except` and continued on failure. Two of
the eleven configured URLs — the bell schedule and clubs pages — had been
year-scoped (`24-25-bell-schedules`, `clubs-organizations`) and started
returning 404 when the school rolled the site over. The corpus shipped without
them and nothing said so.

Both were repointed (`26-27-bell-schedules`, `clubs`), and the scraper now
reports any page that yields no chunks. Restoring them is the single largest
accuracy gain in this repo: **top-3 retrieval 53.3% → 66.7%**.

## Performance

Median over 5 real questions, measured end to end by `eval/latency.py`:

| Stage | Median | Range |
| --- | --- | --- |
| Embed question (ada-002) | 327 ms | 285–797 |
| Retrieve top 3 (cosine, in process) | <1 ms | 0–4 |
| Time to first token (gpt-4o) | 430 ms | 372–1454 |
| Full answer stream | 1,154 ms | 782–2,160 |
| Grounding check (after last token) | 701 ms | 480–845 |
| **End to end** | **2,153 ms** | 1,552–3,402 |

The number that matters for feel is **757 ms** — embed plus retrieve plus time
to first token, the wait before any text appears. Verification's 701 ms lands
*after* the last token, so the full accuracy layer costs nothing in perceived
latency. That is the entire reason it runs post-stream.

Retrieval being under a millisecond is worth noting: 100% of user-visible
latency is network round trips to OpenAI. Optimising the search would buy
nothing.

## What is not measured

Stated plainly, because an unlabeled claim is not a result.

- **Both labeled sets are small.** 20 grounding cases and 25 retrieval
  questions. The LLM judge scoring a perfect F1 on 20 cases means it did not
  fail *here*, not that it does not fail. The injected false claims are also
  fairly blatant; subtle unsupported claims are the harder and untested case.
- **The gate margin is thin.** Answerable questions bottom out at 0.783 top-1
  cosine and out-of-scope questions top out at 0.784 — they now overlap by one
  question, which is why separation is 96% and not 100%.
- **Thresholds are specific** to `text-embedding-ada-002` on this corpus and do
  not transfer to another embedding model.
- **Cost figures are computed** from published per-token rates and token counts,
  not read off a billing dashboard.

### Known weaknesses

Retrieval, not hallucination detection, is the bottleneck. A third of answerable
questions still never retrieve the right page, and the verification layer cannot
repair an answer built on the wrong context — it can only flag it. The causes
are visible in the eval output: fixed 150-word chunks split content mid-topic,
and pure dense retrieval has no term-matching fallback for near-exact queries
like "bell schedule".

Ranked by expected value:

1. Hybrid retrieval (BM25 + dense) — the remaining misses are largely lexical.
2. Semantic chunking on headings instead of a fixed 150-word window.
3. A larger grounding set with subtler hallucinations, to find where the judge
   actually breaks.
4. Scheduled re-scraping, so a year rollover cannot silently empty a page again.

## Reproducing the numbers

Every figure above comes from a script in `eval/`, with raw output committed in
`eval/results/`.

```bash
python eval/corpus_stats.py           # corpus + threshold analysis, no API key
python eval/link_grounding_eval.py    # link check precision/recall, no API key
python eval/retrieval_eval.py         # retrieval accuracy + gate calibration
python eval/detection_ablation.py     # the four detector architectures
python eval/latency.py                # per-stage timings

# point the corpus-based evals at either index
VRHS_EMBEDDINGS=data/vrhs_embeddings_baseline.json python eval/retrieval_eval.py
```

## Layout

| Path | Purpose |
| --- | --- |
| `main.py` | Flask app: scraping, embedding, retrieval, streaming `/ask` |
| `hallucination.py` | The three checks and the notice they produce |
| `corpus.py` | Boilerplate stripping and dedupe, run before embedding |
| `eval/` | Harness, labeled fixtures, and committed raw results |
| `eval/detectors.py` | All four detection architectures, including the unshipped ones |
| `data/vrhs_embeddings.json` | Production index (44 chunks) |
| `data/vrhs_embeddings_baseline.json` | Same scrape without the hygiene pass (108 chunks) |
| `docs/` | README diagram and screenshots |
