# VRHS Chatbot

A retrieval-augmented chatbot for Vista Ridge High School. It answers student
and parent questions from content scraped off the school site, streams the
answer token by token, and then checks its own output for hallucination before
asking anyone to trust it.

Built by Saif Syed and Junayd Elhassan.

```
scrape 11 pages
      |
      v
corpus hygiene ---- strip cross-page nav, drop duplicates
      |
      v
embed (ada-002) --> data/vrhs_embeddings.json
      |
      v
question --> embed --> cosine over all chunks --> top 3 + similarity stats
      |
      v
gpt-4o streams the answer to the browser
      |
      v
hallucination layer: retrieval gate | link grounding | claim grounding
      |
      v
warning appended if any check fails
```

## Why there is a verification layer at all

The failure mode of a small RAG system is not wild invention. It is a fluent,
confident answer built on retrieved text that does not actually support it. For
a school chatbot the concrete version of that is an invented URL: a student
follows `vrhs.leanderisd.org/lunch-menu`, hits a 404, and stops trusting the
tool. So the layer checks three things, cheapest first:

| Check | Method | Cost |
| --- | --- | --- |
| Retrieval gate | Is the best chunk similar enough that the corpus plausibly covers this question? | free |
| Link grounding | Does every URL in the answer appear verbatim in the retrieved context? | free, exact |
| Claim grounding | A second model reads only the context and the answer and reports unsupported claims | one `gpt-4o-mini` call |

Checks are advisory. They never suppress an answer; they append a warning. That
is a deliberate choice driven by the streaming UI: tokens are already on the
reader's screen by the time the answer can be evaluated as a whole, and
retracting text someone just read is worse than annotating it.

The grader is `gpt-4o-mini`, not the `gpt-4o` that wrote the answer. A grader
that shares the generator's failure modes will happily ratify them.

## Engineering decisions, and where measurement overruled me

### Retrieval thresholds: wrong twice before being measured

This is the decision I got wrong most often, so it is the one worth writing down.

**First attempt - intuition.** Cosine cutoffs of 0.82 (solid) and 0.76 (weak),
picked from a general sense of where ada-002 similarity sits. No evidence.

**Second attempt - corpus analysis said absolute cutoffs were impossible.**
Measuring all 4,095 chunk pairs in the committed corpus:

| Statistic | Value |
| --- | --- |
| Median cosine between two *unrelated* chunks | 0.8817 |
| 5th percentile cosine of a *perfect* match | 0.8798 |

The distributions overlap. On that evidence no single cutoff can separate a real
match from noise, so I replaced the threshold with a scale-free standard score,
`(top - mean) / std`, over the query's similarities to every chunk.

**Third attempt - real questions reversed it again.** Step 2 used chunks as
stand-in queries, and chunks are the one thing carrying the site's navigation
boilerplate, which is exactly what inflated that 0.88 baseline. A real question
is short and carries none of it. Measured over 25 real questions against the
cleaned corpus:

| Signal | Best cutoff | Separation accuracy |
| --- | --- | --- |
| Raw top-1 cosine | 0.782 | **100%** |
| Standard-score margin | 2.12 | 84% |

The same comparison on the raw 91-chunk corpus gives raw cosine 100% at a 0.774
cutoff versus 76% for the margin, so the conclusion holds either way.

Answerable questions bottom out at 0.783 top-1 cosine; out-of-scope questions
top out at 0.782. The absolute threshold I had discarded was the right answer;
my proxy experiment was measuring the wrong thing. The gate is now absolute
cosine at 0.78 / 0.77, and the standard score is kept only as a logged
diagnostic.

The lesson recorded in the code: a proxy measurement can be worse than no
measurement, because it is persuasive.

### Corpus hygiene: 72.6% of the corpus was navigation

A naive full-page scrape embeds the site's nav menu into every chunk:

| Property | Value |
| --- | --- |
| Words inside shingles repeated across 60%+ of pages | 8,883 / 12,236 (**72.6%**) |
| Exact duplicate chunks | 15 of 91 |
| Chunks after stripping and dedupe | 91 to 36 (72.5% of words removed) |

`corpus.py` strips cross-page shingles before embedding. Honest accounting of
what that bought, measured by re-embedding the cleaned corpus and re-running the
same 25 questions:

| Metric | Raw corpus (91 chunks) | Cleaned corpus (36 chunks) |
| --- | --- | --- |
| Top-1 source accuracy | 46.7% | **53.3%** |
| Top-3 source accuracy | 53.3% | 53.3% |
| Gate separation (raw cosine) | 100% | 100% |

Top-1 improved; **top-3 did not move at all**. The real win is a 60% smaller
index and 72% fewer embedded tokens per rebuild, not accuracy. Claiming this as
a retrieval improvement would overstate it.

### Link grounding: the obvious regex missed a fifth of the cases

The first version only matched markdown links. Against 16 labeled cases
containing 11 fabricated URLs:

| Version | Precision | Recall | F1 |
| --- | --- | --- | --- |
| Markdown links only | 1.000 | 0.818 | 0.900 |
| Markdown + bare URLs | **1.000** | **1.000** | **1.000** |

The misses were bare URLs dropped into prose, which the model does often enough
to matter. Precision is 1.000 by construction - the check is exact string
containment, so it cannot flag a URL that is actually present.

### Ingest coverage: two pages were silently empty

The scraper wrapped each page in `try/except` and continued on failure. Auditing
the shipped corpus against the configured URL list:

**2 of 11 configured pages produced zero chunks** - the bell schedule and the
clubs/organizations pages. The chatbot did not know things the site documents,
and nothing in the pipeline said so. The scraper now reports pages that yield
nothing instead of swallowing the gap.

This also invalidated part of my own eval: 2 of 15 "answerable" questions
referenced those missing pages and could never have been retrieved. Excluding
them, top-3 accuracy is 8/13 (**61.5%**) rather than 53.3%.

## Measured effectiveness

Everything below is reproduced by the scripts in `eval/`, with raw output
committed in `eval/results/`.

| Result | Value | Source |
| --- | --- | --- |
| Gate separation, answerable vs out-of-scope | 100% (25 questions) | `retrieval_eval.py` |
| Link grounding precision / recall | 1.000 / 1.000 (16 cases) | `link_grounding_eval.py` |
| Top-3 retrieval accuracy | 53.3%, or 61.5% excluding unscrapeable pages | `retrieval_eval.py` |
| Corpus reduction from hygiene pass | 91 to 36 chunks, 72.5% of words | `corpus_stats.py` |
| Boilerplate share of raw scrape | 72.6% of words | `corpus_stats.py` |

### What is not measured

Stated plainly, because an unlabeled claim is not a result:

- **Claim grounding is unevaluated.** The `gpt-4o-mini` grounding pass has no
  labeled dataset behind it. I have not measured its precision or recall, and I
  do not claim a number for it. Building that set is the next piece of work.
- **25 questions is a small calibration set.** The 0.78 threshold separates them
  perfectly, but the margin on the cleaned corpus is 0.001 (0.783 vs 0.782),
  which is too thin to be confident in. It needs a larger question set.
- **Thresholds are specific** to `text-embedding-ada-002` on this corpus and do
  not transfer to another embedding model.

### Known weaknesses

Retrieval, not hallucination detection, is the bottleneck. Roughly 40% of
answerable questions never retrieve the right page, and the verification layer
cannot fix an answer built on the wrong context - it can only flag it. The
causes are visible in the eval output: 150-word fixed-size chunks split content
mid-topic, and pure dense retrieval has no term-matching fallback for queries
like "bell schedule" that should be near-exact matches.

Ordered by expected value:

1. Rebuild the corpus after the ingest fix so the two missing pages exist.
2. Hybrid retrieval (BM25 + dense) - the misses are largely lexical.
3. Semantic chunking on headings instead of a fixed 150-word window.
4. A labeled set for claim grounding, so the third check has a number.

## Running it

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
python main.py                 # serves on :8080
curl localhost:8080/embed      # build the knowledge base (needed once)
```

## Evaluation

```bash
python eval/corpus_stats.py          # offline, no API key
python eval/link_grounding_eval.py   # offline, no API key
python eval/retrieval_eval.py        # needs OPENAI_API_KEY (~25 embedding calls)

# point the retrieval eval at either corpus
VRHS_EMBEDDINGS=data/vrhs_embeddings_clean.json python eval/retrieval_eval.py
```

## Layout

| Path | Purpose |
| --- | --- |
| `main.py` | Flask app, scraping, embedding, retrieval, streaming `/ask` |
| `hallucination.py` | The three checks and the warning they produce |
| `corpus.py` | Boilerplate stripping and dedupe, run before embedding |
| `eval/` | Evaluation harness, labeled fixtures, committed raw results |
| `data/vrhs_embeddings.json` | Original corpus (91 chunks) |
| `data/vrhs_embeddings_clean.json` | After the hygiene pass (36 chunks) |
