# VRHS Chatbot

A retrieval-augmented chatbot for Vista Ridge High School. It answers questions
from the school's own web pages, streams the answer as it is written, shows
which pages it drew from, and checks its own output for hallucination before
asking anyone to trust it.

Built by Saif Syed and Junayd Elhassan. The live deployment is linked in the
repository sidebar.

![The chatbot answering a question, with source pills and a verification note](docs/screenshot.png)

## What it does

* Answers from 384 chunks covering all 23 pages of the school site and the 17
  linked Google Docs it reads, not from model memory. Pages are found by
  crawling, so a year rollover does not break it and nothing has to be added to
  a list by hand.
* Streams tokens, so text appears after about 660 ms.
* Attaches **source pills** linking to the pages behind the answer.
* Runs three grounding checks and posts a short caution when one fails.
* Records thumbs ratings against retrieval scores, so gaps become visible.

## Architecture

![Question, retrieval, streaming, then three verification checks](docs/architecture.svg)

The index is a JSON file of 384 vectors held in memory. At this size a vector
database would add operational weight and no speed: brute-force cosine over 384
vectors takes under a millisecond, which is three orders of magnitude below the
network round trip in front of it. The bottleneck is API latency, not search.

### Pages are crawled, not listed

This took two attempts, and the first was still a whitelist.

Originally eleven URLs were named directly. Two were year-scoped
(`24-25-bell-schedules`) and started returning 404 the moment the school rolled
the site over, silently dropping those pages. Naming `26-27` instead would have
failed the same way next summer, so discovery replaced it: read the live
navigation, follow links whose label or path matches a topic worth having.

That was better and still wrong. A topic list only finds what somebody thought
to name. Asked when doors open for the Saturday SAT, the bot had nothing,
because no pattern matched `/saturday-sat-test` even though the homepage links
it directly. Nine pages were missing for that reason, including the principal's
office and the staff page.

Ingest now crawls the site: breadth-first from the homepage, same domain only,
skipping binary files, bounded at 40 pages and depth 2. It finds all 23 pages in
about twenty seconds, and removes the entire class of gap rather than the
instance of it.

### Links live inline, and each one is also its own chunk

The scraper used to append every link to an `Important Links` block at the end
of the page text. Chunking then split that block away from the prose that
explained it, producing chunks of bare URLs that matched nothing. Asking "where
can I find bus information" retrieved the paragraph naming *Bus Routes / Smart
Tag* at rank 1, while the chunk holding the only bus URL on the site sat at
**rank 27 of 44**. The answer named the page and could not link to it.

Two changes fixed it. Links are now inlined as `[label](url)` inside the
sentence that gives them meaning, and every unique link additionally becomes
its own small chunk keyed on its label. A page like `parent_resources` is
mostly a list of twenty links; at 150 words per chunk that was one blob
covering twenty unrelated topics, and its embedding was a blur that matched
none of them. Giving each link its own chunk makes the label the thing being
matched, which is what a "where do I find X" question is actually asking.

The same question now retrieves *Bus Info* at rank 1, and **100% of answerable
questions retrieve a context containing a usable link**, measured across the
eval set.

### Prose and links are retrieved under separate quotas

Link chunks outnumber prose chunks roughly two to one, so a plain top-3 fills
with links. Asked who the principal is, retrieval returned his name as a link
label at rank 2 with nothing stating he holds the job, while the paragraph that
said so sat at rank 6. The model then correctly declined to answer a question
the corpus could answer.

Retrieval now fills three prose slots and two link slots separately, so an
answer and somewhere to go both make it into the context. Top-3 source accuracy
went 60.0% to 66.7%, and the retrieval gate went from 96% to **100%**
separation on the question set. `eval/retrieval_eval.py` selects under the same
quotas, because measuring a plain top-k while the app runs quotas would report
numbers describing a system nobody is running.

### The site links what it does not say

The crawl is same-domain and skips binary files, which meant the answers to
some of the most-asked questions were one hop away and invisible. Bell times
are not on the website; they are in a Google Doc the website links to.

Two fetches now happen at ingest, and they fix different things.

**Document text.** A public Google Doc exports as plain text from a URL, with
no API key and no credentials. 17 of the 26 linked documents come back with
real content and become chunks like any page. Asked what time school starts,
the bot used to hand over a link; it now answers **8:15 AM, and 10:15 on late
start days**, because it has read the schedule.

**Drive filenames.** The site links two files both labelled `A/B Calendar`, one
of them last year's, and nothing in the page text tells them apart — so the
model chose blind and chose wrong, which is exactly the complaint that started
this. The filename settles it:

| File | Real filename |
| --- | --- |
| `1xX13…` | 2025-2026 District A_B Calendar.pdf |
| `1n54N…` | **VRHS 2026-2027 Calendar.pdf** |

That arrives in a `Content-Disposition` header, on a request that does not have
to finish downloading. The label becomes `A/B Calendar (VRHS 2026-2027
Calendar)`, the existing preference for year-bearing labels does the rest, and
the bot now links the current calendar.

Worth noting what was *not* needed. The obvious approach was reading the PDFs,
and the one that would have required OCR — an image-only export with no text
layer — turned out to be precisely the one whose filename already said what it
was. The corpus went from 279 chunks over 23 pages to 384 over 41 sources.

### A link chunk belongs to what it points at

Link chunks recorded the page the anchor was *found* on. Most links live in
site-wide navigation, so that attribution was close to arbitrary: asked where
the library website is, the answer cited **Saturday SAT Test** — a page that
says nothing about the library and merely carries the same nav bar. The reader
is sent to a source that cannot confirm the answer.

A link chunk is *about* its target. That is what it says, what it matched on,
and what a reader following the pill wants. Source is now the target URL and
the pill carries the site's own wording for the link, which also beats deriving
a name from the URL — `source_label()` called the library "Vrhslibrary".

Free to fix on the existing index: `source` is not part of the embedded text,
so 250 link chunks were re-pointed without re-embedding anything. The page an
anchor was found on is kept as `found_on`, because a wrong link is easier to
chase when you know where it came from.

### Follow-up questions

Every request carried only the current line. Asked *"Does Dr. Morgan go to this
school?"* and then *"Who is he?"*, the bot answered the second cold — the
pronoun had nothing to refer to, and retrieval matched three function words
against the corpus:

> Based on the context provided, "he" could refer to…

The client now keeps the last three exchanges and sends them, and they sit
between the system prompt and the current turn. The context block stays
attached to the current question rather than being sent as its own turn, so an
earlier turn's context is not treated as still in force.

Retrieval needs separate handling, because *"Who is he?"* embeds to nothing
useful. When a question looks dependent — four words or fewer, or carrying a
pronoun — the previous user turn is prepended before embedding. Concatenation
rather than a model call to rewrite the question: rewriting would sit in front
of retrieval, which sits in front of the answer, adding a round trip to the one
path a reader is actually waiting on. Gluing two questions together is free and
puts the missing subject back. Only for dependent-looking questions, since
concatenating unconditionally would blur a genuine change of subject.

State lives in the browser because the server has no sessions, which is what
lets it run on one free instance with several threads and no store.

**One caveat, and it is a real one.** A follow-up can now be answered from the
conversation rather than from the corpus. Asked what the principal does, the
bot answered correctly from what it had already said, with retrieval scoring
`none` and no sources shown — which is honest, but it means history has opened
a path where the retrieval gate has nothing to grade. Cached answers are
deliberately not served to follow-ups for the same reason: the cache is keyed
on the question alone and knows nothing about what came before.

## The hallucination layer

The failure mode of a small RAG system is not wild invention. It is a fluent
answer built on retrieved text that does not support it. The concrete version
here is an invented URL: a student follows a link, gets a 404, and stops
trusting the tool.

Before any of that, the prompt tries to stop the problem happening. The model
is told to answer only from the retrieved context, never to infer times, dates,
fees, names or requirements the context does not state, never to construct a
URL, and to say plainly that it does not know rather than assemble something
plausible.

A second instruction adds today's date and the school year derived from it.
The site carries more than one year at once: the calendar page still links a
2025-2026 district calendar while the navigation already points at 26-27 bell
schedules. With no sense of the date the model could not tell which was
current, and presented last year's calendar as this year's. It now names the
year a page refers to, and says when the current one is missing:

> I couldn't find the district calendar for the 2026-2027 school year. However,
> you can check the 2025-2026 LISD District Calendar.

An earlier version also appended a warning whenever the retrieval score was
weak. It was removed: it made answers hedge more without making them more
accurate, and the three checks below already cover that case after the fact.

Prevention is cheap but not sufficient, since a model told to abstain still
sometimes does not. The checks assume it failed.

Three checks run on every answer, cheapest first.

| Check | Question it answers | Cost | Measured |
| --- | --- | --- | --- |
| Retrieval gate | Does the corpus cover this at all? | free | 100% separation, answerable vs out of scope |
| Link grounding | Does every URL appear verbatim in the context? | free | precision 1.00, recall 1.00 |
| Claim grounding | Does the context support each claim? | $0.00008 | precision 0.80, recall 0.89 |

Two decisions worth stating.

**Checks annotate, they never retract.** Verification can only run once the
answer exists as a whole, and by then the tokens are on screen. Deleting text
someone just read is worse than flagging it.

**A caution is for confident, not for wrong.** The retrieval warning used to
fire on any weak match, including on replies that had already said they did not
know. Asked whether tomorrow is an A day, the bot would answer that it could
not tell and point at the calendar, and *Worth double-checking — only a loose
match on the school pages* would appear underneath. That says the same thing
twice in a more doubtful voice, and it reads as the bot being unsure of its own
honesty.

The warning exists to catch misplaced confidence. A reply that declined has
none to misplace, so the retrieval notice is now suppressed when the answer
gave up — and only then. If the model asserted something on a weak match, the
warning is exactly what it was built for and it stays. A fabricated link or an
unsupported claim keeps its notice whatever the tone, because those are
concrete problems rather than questions of confidence.

Whether a reply declined is decided by a model rather than a phrase list, for
the same reason as everywhere else here: *"I don't have the exact time"* inside
an otherwise complete answer is not a refusal, and *"that isn't something the
school pages cover"* is one with none of the obvious markers. The check runs
only when it could change the outcome — retrieval short of solid, and both
concrete checks already clean — so most answers never pay for it. It measured
the caution rate 24.0% → **16.0%** with nothing reaching the reader unflagged.

**The grader is not the generator.** Claim grounding runs on `gpt-4o-mini`
while answers come from `gpt-4.1`. A grader that shares the generator's blind
spots will approve its mistakes.

When a check fails, the answer keeps its text and picks up a short note. The
panel is capped at two points, because an earlier version listed every
unsupported claim and buried the answer in hedging.

![A question the corpus does not cover, with the verification note](docs/verification.png)

Sources are withheld when the gate says the corpus does not cover the question.
Citing a page that did not support the answer is its own kind of false
confidence.

## Picking a detection architecture

The LLM judge is the slowest and most expensive option, so it needed to earn
the slot. Five designs were built and measured against 20 labeled cases, 9 with
injected false claims and 11 grounded, drawn from real chunks in the corpus.

| Architecture | Precision | Recall | F1 | p50 latency | $/check |
| --- | --- | --- | --- | --- | --- |
| Lexical overlap | 0.364 | 0.889 | 0.516 | 0 ms | $0 |
| Embedding similarity | 0.348 | 0.889 | 0.500 | 339 ms | $0.000022 |
| Cascade, lexical then LLM | 0.364 | 0.889 | 0.516 | 0 ms | $0.000018 |
| Chain-of-Verification | 0.667 | 0.889 | 0.762 | 1,376 ms | $0.000162 |
| **LLM judge (`gpt-4o-mini`)** | **0.800** | **0.889** | **0.842** | 890 ms | $0.000084 |

These are the 33-case set. On the original 20 the top three scored 0.889, 0.941
and 1.000, and those numbers were flattering everything: the fixture contained
only substantive claims, so a detector was never asked to leave a refusal
alone. Adding 13 real declining replies took lexical overlap from 0.889 to
0.364. It had not got worse; it had always done this, and nothing had been
looking.

What the numbers say:

**Chain-of-Verification lost, and that is the most useful result here.** CoVe
(Dhuliawala et al., 2023) is a published anti-hallucination method: plan
verification questions about the answer, answer them independently, then judge.
It scored precision 0.615, the worst of any option, at double the cost and
nearly three times the latency. The reason is diagnosable. Decomposition
generates questions about incidental details, and requiring the source to state
each one turns ordinary paraphrase into a false positive. Loosening the prompt
to accept paraphrase moved precision to 0.636 but cost recall, for a slightly
worse F1 of 0.700. It stays in `eval/detectors.py` as a measured alternative
rather than shipping because it sounds rigorous.

Its score also moves between runs more than anything else in the table —
precision has come out at 0.615, 0.667 and 0.692 on the same 20 cases, because
two chained model calls compound their own variance. The row above is one
snapshot, and `eval/results/detection_ablation.txt` holds the most recent. Its
ranking has never changed, which is the part the decision rests on.

**Lexical overlap is a strong baseline.** F1 0.889 for nothing. Any argument
that a model call is necessary has to beat this first.

**Embedding similarity was the worst of the cheap options.** Sentence vectors
stay close to the context whenever the topic matches, and an injected false
claim usually keeps the topic. It fails exactly where it is needed.

**The cascade is the one to watch.** A lexical prefilter resolves the confident
cases and escalates only the ambiguous band, so 13 of 20 cases never made an API
call: F1 0.941 at a third of the cost and a 0 ms median.

**The judge ships** because at this volume the cost gap is meaningless. At
10,000 questions a month the judge costs $0.72 against the cascade's $0.25.
Paying $0.47 to recover the last 6 points of F1 is obviously right. The cascade
becomes the correct choice at roughly 100 times this traffic.

The two threshold-based detectors are reported at their best achievable F1, with
the threshold chosen on the same 20 cases that score them. That flatters them,
and the judge still wins.

## How often does it actually hallucinate?

Everything above measures the detector. Twenty labelled cases with false claims
injected by hand say how good the judge is *once a hallucination exists*, and
nothing at all about how often one exists, because none of those answers were
written by this bot.

`eval/hallucination_rate.py` asks the other question. It imports `main.py` and
calls the same `get_relevant_context` and the same `dated_prompt` the Flask
route calls, at the same model and the same default temperature, then runs the
same `verify_answer` over the result. Every question runs twice, because
temperature 1.0 means one sample is an anecdote.

Fifty trials on `gpt-4.1`, 30 on questions the corpus covers and 20 on
questions it does not.

| Rate | Result | |
| --- | --- | --- |
| Fabricated link | 0 of 50 | **0.0%** |
| Unsupported claim | 8 of 50 | 16.0% |
| Answered a question the site does not cover | 0 of 20 | **0.0%** |
| Refused a question the site does cover | 3 of 30 | 10.0% |

And the number that matters more than any of those:

| What the reader sees | Result | |
| --- | --- | --- |
| Trials with any fault | 8 of 50 | 16.0% |
| Of those, flagged to the reader | 8 of 8 | **100%** |
| Of those, reached the reader unflagged | 0 of 8 | **0.0%** |
| Answerable, no fault, warned anyway | 0 of 30 | **0.0%** |

Nothing got through unflagged, and nothing clean got warned. The two failures
the layer exists to prevent, a silent hallucination and a caution the reader
learns to ignore, both came in at zero on this set.

### The model swap was not free, and this is where it was paid

Moving from `gpt-4o` to `gpt-4.1` for the 211 ms above was measured on the same
harness before it shipped. It is not a clean win:

| | `gpt-4o` | `gpt-4.1` |
| --- | --- | --- |
| Fabricated link | 0.0% | **0.0%** |
| Answered an uncovered question | 0.0% | **0.0%** |
| Reached the reader unflagged | 0.0% | **0.0%** |
| Refused a question the corpus covers | 23.3% | **15.6%** |
| Unsupported claim | 24 to 28% | **38.7%** |

The first check was whether 34% on the first 50-trial run was noise, since
`gpt-4o` had itself ranged 24 to 28% across three runs. It was not: at 75
trials it went to 38.7%, further from `gpt-4o`, not closer.

Both columns were measured with the single-call grader, before it was split
into a detector and a filter. That change later took `gpt-4.1` from 38.7% to
24.0%, so the gap above is real but the absolute numbers are historical.
`gpt-4o` has not been re-measured under the new grader, and the comparison is
kept as it was taken rather than half-updated.

The mechanism is coherent rather than mysterious. `gpt-4.1` is less
conservative, so it refuses a third fewer of the questions the corpus can
answer *and* makes more claims the grader will not certify. Those are the same
disposition seen from two ends. Part of the rise is genuinely more unverified
content and part is simply more content to grade, and this harness cannot
separate them — a bot that says less has less to be wrong about.

What decided it is that the three failures a reader cannot defend against all
held at zero. No invented links, nothing answered from pretraining, nothing
slipping past unflagged. What rose is the rate of answers carrying a visible
caution, which is the layer doing its job in public. Trading silent risk for
visible hedging is the right direction for a tool students are told to trust.

`VRHS_CHAT_MODEL` sets this, so reverting is an environment variable rather
than a deploy.

### The grader was flagging the bot for refusing

The first run of this harness reported a 66% unsupported-claim rate, which was
wrong in an instructive way. On out-of-scope questions, where the bot correctly
refuses, **18 of 20 refusals were flagged as containing an unsupported claim**.
The flagged text was the refusal:

> "I don't have the information about the plot of Hamlet."
> "I'm here to help with inquiries related to Vista Ridge High School."
> "contact the front office for assistance."

The grader was right that the context does not state those things. It was asked
the wrong question. A sentence about what the assistant knows is not a claim
about the world, and grading it against a scrape of the school website is a
category error. The prompt already said to ignore "hedged suggestions to
contact the school" and the grader ignored it anyway, because a one-line
exclusion phrased as a category does not survive contact with a concrete
sentence.

The cost was not academic. Every one of those refusals shipped with *"Some
details are not confirmed by the pages I read"* attached, so the caution panel
fired hardest on the answers behaving best — training readers to dismiss the
panel exactly where the real warnings live.

Naming the exclusions concretely, and leading with the distinction that carries
the weight — claims about the school, not claims about the assistant — took the
rate from **66% to 24%**, with the judge still scoring precision 1.00 and
recall 1.00 on the 20 labelled cases. The looser prompt cost no recall.

### The benchmark could not see the bug, so four fixes went unscored

A report from real use: the caution *"Some details are not confirmed by the
pages I read"* was appearing far too often. Auditing the 62 flagged claims in
`eval/results/hallucination_rate_trials.json` said the complaint was right and
worse than it looked — **54 of 62 were not claims at all**:

| What was flagged | Count | |
| --- | --- | --- |
| Referrals — *"contact the front office"* | 23 | 37% |
| Bare noun phrases naming what was **missing** — *"senior checklists"* | ~14 | 23% |
| Refusals — *"the context does not list..."* | 8 | 13% |
| Sign-offs — *"Have a great September!"* | 5 | 8% |
| **Actual claims about the school** | **~6** | **10%** |

The grader was answering a different question: what does the answer *discuss*
that the source does not cover, rather than what does the answer *assert* that
the source does not support.

Three fixes by instruction failed. Naming the exclusions, naming them again
with examples, then demanding verbatim quotes with worked cases, moved the flag
rate 38.7% → 34.7%. The grader kept reporting the same sentences and merely
quoted them more accurately.

Restructuring it as *label every sentence, then report only what you called a
fact* fixed precision outright — 38.7% → **5.3%** — and dropped recall from
1.000 to **0.556**. That is the worse error by a distance, and it exposed the
real problem:

**The 20-case fixture could not score any of this.** Every case in it was a
substantive claim. It contained no refusals, no referrals, no sign-offs — so a
change that filters out non-claims could only ever look worse on it. The cost
landed on recall; the benefit was invisible. Four attempts had been tuned
against an instrument blind to the thing being fixed.

So the fixture was extended with 13 real declining replies from the trials,
labelled *must not be flagged*. The correct label needs no judgement: a reply
that declines asserts nothing about the school. On that 33-case set:

| Grader | Precision | Recall | F1 | Refusals wrongly flagged |
| --- | --- | --- | --- | --- |
| Original, one call | 0.600 | **1.000** | 0.750 | 3 of 13 |
| **Detector + filter** | **0.800** | 0.889 | **0.842** | **0 of 13** |

Two calls, because one model asked to do both jobs did each of them worse. The
detector runs unchanged and stays suspicious; a second pass decides which of
its findings were claims at all, sees only the sentences and never the source,
and runs only when the detector found something — so a clean answer still costs
one call, and both are post-stream where latency never reaches the reader.

It costs one missed hallucination in nine. It buys the elimination of the
false cautions on this set and takes the production rate 38.7% → **24.0%**.

One more attempt is recorded in the code and was reverted: a regex keeping any
claim carrying a name, email, time or date regardless of the filter's label,
written for *"emailed directly to the principal, Dr. Keith Morgan"* — a
referral in form, a fabrication in substance. Precision fell 0.800 → 0.727 and
recall did not move, so it readmitted false positives without recovering the
case it was built for, which means that case is lost somewhere other than the
filter.

### What is left is over-refusal, not invention

The remaining problem is the opposite of the one the layer was built for. The
bot refuses **10.0%** of questions the corpus can answer, and it does so on
questions where retrieval had already succeeded:

| Refused question | Top cosine | Gate |
| --- | --- | --- |
| Where do I drop off my student in the morning? | 0.839 | solid |
| Where do I report an absence? | 0.828 | solid |
| What do seniors need to do before graduation? | 0.802 | solid |
| How do I find a teacher's email? | 0.795 | solid |

Retrieval is not the fault here — the gate says solid every time, and the right
page is in the context. The model is reading a context that contains a relevant
link and some surrounding prose, and deciding that is not enough to answer
with. That points at the prompt rather than the index: the instruction not to
infer beyond the context is doing more work than intended, and a link chunk
whose prose says little may not read as permission to answer.

That is the next thing to fix, and it is worth saying that it was invisible
until this harness existed. Every metric in this repo before it measured
whether the bot says false things. None measured whether it says anything.

## Feedback loop

Every answer carries a thumbs rating, and a thumbs-down opens three reasons:
not found, not up to date, inaccurate. Ratings used to change the on-screen text
and go nowhere. They now POST to `/feedback` and are stored with the question
and the retrieval diagnostics for that answer.

Pairing the two is the point. The rating alone says an answer was poor. With the
retrieval score attached it says why, and the two failures need different fixes:

| Pattern | Reading | Fix |
| --- | --- | --- |
| Thumbs-down, retrieval below the gate | The site does not cover this | Admin adds a page, or it becomes a manual entry |
| Thumbs-down, retrieval solid | Right page found, answer still poor | Prompt or chunking problem |
| Thumbs-down on a flagged answer | The layer already caught it | Working as intended |

### Where the feedback goes, and why it needs a private repo

Render's filesystem is ephemeral, so `data/feedback.json` is wiped on every
restart and redeploy. To keep ratings, set `VRHS_GITHUB_TOKEN` to a
fine-grained token with **Issues: read and write** on one repository, and each
thumbs-down opens an issue there. A thumbs-up does not: it needs no triage, and
filing one would bury the ones that do.

Point `VRHS_GITHUB_REPO` at a **private** repository. The question and the
comment are the only two fields a reader writes, and they are the only two that
can carry a name, a student id or a sentence about a named teacher. The server
checks the destination against the GitHub API before it files anything, and if
the repository is public — or if the check fails, or the token cannot see it —
it withholds both and files the diagnostics alone. That is deliberately close to
useless, because an unusable issue is better pressure toward a private tracker
than a paragraph here asking nicely.

Both fields travel inside a code fence, so an `@name` a reader types cannot
notify a real person and a `#12` cannot cross-link onto an unrelated issue. The
fence grows longer than any run of backticks in the text, and invisible
characters — the C0 range, zero-width marks, the bidirectional overrides — are
stripped first.

`/feedback` is public and unauthenticated, so it is capped at
`VRHS_GITHUB_ISSUES_PER_HOUR` issues an hour (12 by default). Past the cap
ratings still reach the log and the file. Request bodies are capped at 256 KB
across the whole app: field truncation happens after parsing, so without that
limit a single large POST is read into memory in full.

## What changed and what it bought

| Change | Measured effect |
| --- | --- |
| Discovered pages from the live navigation instead of a hardcoded list | 2 URLs that 404'd on rollover fixed permanently, 14 pages found |
| Inlined links and gave each one its own chunk | bus URL **rank 27 to rank 1**, link-in-context **100%** |
| Extended link checking to bare URLs, not just markdown | recall **0.818 to 1.000** |
| Absolute cosine gate instead of a standard-score margin | separation **84% to 96%** |
| Corpus hygiene before embedding | index **263 to 218 chunks**, words **11,710 to 4,203**, top-3 up 6.7 points |
| LLM judge over the alternatives | F1 **0.727 to 1.000** against CoVe, for less cost |
| Verify after the last token, not before the first | perceived wait unchanged at **757 ms** |
| Warmed the index and the API connection at boot | **0.8 to 1.3 s** off the first question after a cold start |
| Cached query embeddings on the normalised question | repeat questions skip a **218 ms** round trip |
| Told the grader that refusals are not claims | unsupported-claim rate **66% to 24%**, judge F1 unchanged at 1.00 |
| Moved the answering model to `gpt-4.1`, measured interleaved | TTFT **-211 ms**, over-refusal **23.3% to 15.6%**, claims 24% to 38.7% |
| Held the API connection open past httpx's 5 s default | **-78 ms** on a bot asked a question every few minutes |
| Replaced the Flask dev server with gunicorn | a production WSGI server, worker supervision, streaming-safe timeouts |
| Moved every literal into `config.py`, read from the environment | model and thresholds change without a redeploy |
| Pinned every dependency | a redeploy installs what the measurements were taken against |
| Pre-generated and pre-verified answers for common questions | **660 ms to 3-6 ms** server-side, and the checks became preventive |
| Split the grader into a detector and a claim filter | caution rate **38.7% to 24.0%**, refusals wrongly flagged **3 of 13 to 0** |
| Read the Google Docs the site links to | corpus **279 to 384 chunks**, and bell times became answerable at all |
| Took A/B calendar dates from the Drive filename | stopped linking last year's calendar, without parsing a PDF |
| Dropped source pills below the retrieval gate | pills name pages that matched, not whatever filled the quota |
| Stopped cautioning an answer that had already declined | caution rate **24.0% to 16.0%**, nothing reaching the reader unflagged |
| Extended the grounding fixture with 13 real refusals | the benchmark can finally score the failure that shows up in use |
| Index cached in memory instead of re-read per question | one file read and parse per process, not per question |
| Feedback wired to storage | ratings became data instead of a UI state change |

### Two source pages had been silently empty

The scraper wrapped each page in `try/except` and continued on failure. Two of
the eleven URLs were year-scoped (`24-25-bell-schedules`,
`clubs-organizations`) and began returning 404 when the school rolled the site
over. The corpus shipped without them and nothing said so.

They were repointed by hand first, which fixed the symptom and left the cause:
the next rollover would break the same way. Ingest now discovers pages from the
live navigation, so the current bell schedule page is found by what it is
about. The scraper also reports any page that yields no chunks, so a future
gap is loud rather than silent.

### The retrieval threshold was wrong twice

1. **Intuition.** Cutoffs of 0.82 and 0.76, picked from a general sense of where
   ada-002 sits. No evidence.
2. **Corpus analysis said absolute cutoffs were impossible.** Unrelated chunks
   had a median cosine of 0.8920 while a perfect match sat at 0.8802 for the
   5th percentile, measured on the corpus as it stood then. Overlapping, so no single cutoff works. That argued for a
   scale-free standard score.
3. **Real questions reversed it.** Step 2 used chunks as stand-in queries, and
   chunks are the one thing carrying the navigation boilerplate that inflated
   the baseline. Real questions are short and carry none of it.

| Signal | Best cutoff | Separation |
| --- | --- | --- |
| **Raw top-1 cosine** | 0.780 | **96%** |
| Standard-score margin | 2.42 | 84% |

The threshold that had been discarded was correct. The proxy experiment was
measuring the wrong thing. A proxy measurement can be worse than no measurement,
because it is persuasive.

### Three quarters of the corpus was navigation

Both corpora below come from the same scrape, so the comparison is clean.

| | Baseline | After hygiene |
| --- | --- | --- |
| Chunks | 356 | **279** |
| Duplicate chunks | 38 | **0** |
| Words inside cross-page boilerplate | 73.1% | **21.5%** |
| Top-1 retrieval | 40.0% | **46.7%** |
| Top-3 retrieval | 66.7% | 66.7% |
| Answerable questions with a link in context | 100% | 100% |

The pass strips 64% of the words and every duplicate, and on the current corpus
it also gains one question of top-3 recall. On the previous corpus it cost one
instead. At n=15 a single question is inside noise either way, so the honest
reading is that hygiene is a corpus quality and cost win whose effect on recall
is not resolvable at this sample size.

The remaining 22.7% boilerplate figure is an artefact of the link chunks, which
share a short suffix by design. Removing that suffix was tried: it dropped the
background cosine from 0.85 to 0.79 but cost 6.7 points of top-3 retrieval and
raised out-of-scope scores, with gate separation unchanged at 96%. The suffix
stayed, because the background number was cosmetic and the recall was not.

## Performance

Median over 5 real questions, from `eval/latency.py`.

| Stage | Median | Range |
| --- | --- | --- |
| Embed question (ada-002) | 227 ms | 220 to 758 |
| Retrieve under quotas (in memory) | under 1 ms | 0 to 1 |
| Time to first token (gpt-4o) | 518 ms | 467 to 787 |
| Grounding check (after last token) | 713 ms | 583 to 832 |
| End to end | 2,223 ms | 1,862 to 2,583 |

The number that matters is **746 ms**, the wait before any text appears.
Verification's 713 ms lands after the last token, so the accuracy layer costs
nothing in perceived latency. That is the whole reason it runs post-stream.

Retrieval being under a millisecond is worth noting: essentially all
user-visible latency is network round trips to OpenAI, so optimising the search
would buy nothing.

### The table above measures a warm process, and users do not get one

That is the right way to read steady-state cost and the wrong way to understand
why the deployed bot feels slow. The deployment target is Cloud Run, which
scales to zero. The first question after a quiet period lands on a container
that has parsed no index and opened no connection to the API, and none of that
work appears anywhere in the stage table, because the harness did it before
starting the clock.

Measured separately, from a genuinely cold process:

| Cost paid once per container | Measured |
| --- | --- |
| Parse the 9.7 MB index and normalise it | 118 to 137 ms |
| DNS and TLS handshake on the first API call | 629 to 1,154 ms |
| Importing `requests` and `bs4`, which only the scraper needs | ~50 ms |

The handshake is the one that hurts, and it was landing entirely on whoever
asked first. `warm_start()` now does all three at boot on a daemon thread, so
the port opens immediately — which is what Cloud Run watches to decide the
container is ready — and the work finishes while nobody is waiting. Between
**0.8 and 1.3 seconds** comes off the first question, depending on how the
handshake goes.

The import saving deserves a footnote, because the first figure was wrong.
Timed in isolation `requests` and `bs4` cost 525 ms, which is what made moving
them look worthwhile. Timed as a marginal change to this app they cost about
50 ms, because `openai` and `flask` already pull most of their dependency tree
in. The change stayed anyway — it is free and importing a scraper to answer a
question was wrong regardless — but the honest number is 50, not 525.

### Not making the call beats making a faster one

The query embedding is 218 ms sitting directly in front of the answer. Nothing
overlaps it: retrieval cannot start until the vector exists, and the chat call
cannot start until retrieval finishes.

The obvious move is a faster model, and it does not work:

| | Median |
| --- | --- |
| `text-embedding-ada-002` | 218 ms |
| `text-embedding-3-small` | 211 ms |

Seven milliseconds, inside the noise, because the cost is the round trip rather
than the model.

### Trading the model *down* buys nothing. Trading it *across* does

The first pass at this concluded that the answering model did not matter
either, on the grounds that `gpt-4o-mini` reached its first token in 859 ms
against `gpt-4o`'s 882 ms. That conclusion was wrong, and the reason it was
wrong is worth more than the conclusion.

Models were measured one after another, so each one saw whatever the network
was doing when its turn came. On a home connection that varies by more than the
effect being measured — the same embedding call has come back at 167 ms and at
508 ms within the hour. Sequential measurement of a small effect through a
noisy channel does not produce a weak result, it produces a confident wrong
one.

Re-measured with the models **interleaved in randomised order**, so every model
sees the same conditions:

| Model | Median TTFT | Range |
| --- | --- | --- |
| `gpt-4o` | 660 ms | 490 to 2,594 |
| `gpt-4.1-mini` | 566 ms | 528 to 1,301 |
| **`gpt-4.1`** | **448 ms** | **395 to 676** |
| `gpt-4.1-nano` | 421 ms | 357 to 453 |

`gpt-4.1` is **211 ms faster than `gpt-4o`**, and the tail matters as much as
the median: `gpt-4o`'s worst trial was four times its best, while `gpt-4.1`
stayed inside a 280 ms band. The median case never felt broken; the tail is
what a reader notices.

The reasoning models are not candidates and it is not close. `gpt-5-mini`
reaches its first token in 4.3 s and `gpt-5-nano` in 7.8 s, because they think
before they emit. For a streamed answer that is the entire budget.

`gpt-4.1-nano` was faster again by 27 ms, and was not taken. Testing a much
smaller model against the grounding set to save 27 ms is not a trade worth the
evaluation time.

**Read absolute latency here with suspicion, and deltas with confidence.** Every
paired comparison in this section was interleaved. Every unpaired number is one
snapshot of a home connection and moves by a factor of three.

What does work is not making the call. A school chatbot is asked the same
handful of things over and over, so query embeddings are cached on the
normalised question, bounded at 512 entries:

| | Median |
| --- | --- |
| Cache miss, round trip | 183 to 258 ms |
| Cache hit, dict lookup | under 0.01 ms |

A repeat question skips the hop entirely and produces byte-identical context
and statistics. `/health` reports hits and misses, because the hit rate in
production is the only thing that says whether this is buying anything real as
opposed to anything on a benchmark that asks the same question twice.

### Answering before the question arrives

Caching the embedding removes one of the two round trips. The other is the
model, and it cannot be removed — only avoided by having answered already.

`prewarm.py` generates answers for the questions a school gets all year,
checks them properly while nobody is waiting, and writes
`data/answer_cache.json`. `main.py` consults it before retrieval, since there
is no point embedding a question whose answer is written.

Measured against the live path:

| | Server-side | With ~120 ms from a browser to Render |
| --- | --- | --- |
| Novel question | 660 ms | ~780 ms |
| Query-cache hit | ~450 ms | ~570 ms |
| **Pre-generated answer** | **3 to 6 ms** | **~125 ms** |

**This is the one place the checks stop being advisory.** Everywhere else they
annotate an answer the reader has already seen, because an answer is only
checkable once whole and by then the tokens are on screen — the README says as
much under *Checks annotate, they never retract*. A pre-generated answer has no
reader yet, so a variant that fails is discarded rather than shipped with a
caution. The bar is deliberately higher than the live path's: any fabricated
link, any unsupported claim, or a retrieval level below `solid` and the variant
is dropped. If none survive, the question is left to the live path, which is a
perfectly good outcome.

**Variants vary wording, not facts.** Several answers are generated per
question and rotated so the bot does not repeat one phrasing all year. That is
only safe if they agree, and "each passed the grounding check" does not
establish it — two answers can both be supported by the context and still send
a student to different places. Variants must therefore agree on the pages they
cite.

Agreement is by **majority, not unanimity**. Requiring all three to match threw
away whole questions over one odd generation — at temperature 1.0 a model will
occasionally cite an extra page — and rejected 4 of 15 questions that two of
three variants agreed on perfectly well. The odd one out is dropped and the
agreeing ones kept. No majority at all, three variants and three different link
sets, means the question is answered differently every time and goes back to
the live path.

A lone survivor is exempt, which looks like a loophole and is not. The rule
exists to make *rotation* safe; one variant does not rotate, so there is
nothing for it to contradict, and it still had to pass every check the live
path applies and then some.

Run against the 15 answerable fixtures: **9 questions admitted, 16 variants**,
6 left to the live path. Measured end to end through the route:

| | TTFB |
| --- | --- |
| Cached question | **2.9 to 4.0 ms** |
| Same route, uncached question | 930 ms |

**A rebuilt index retires the cache automatically.** The file carries a
fingerprint hashed over the chunk text — not the file, whose mtime changes on
every checkout and which carries 1536 floats per chunk that have no bearing on
whether an answer is still true. On a mismatch `main.py` logs a warning and
ignores the cache entirely rather than serving answers grounded in text nobody
retrieves any more. Verified by pointing a deliberately stale fingerprint at a
live index: 0 variants loaded, warning logged, live path unaffected.

Question sources are the eval fixtures and, when present, `data/feedback.json`
— the questions students actually asked. The second is the better source, and
it is why `/health` reports the hit rate: a list guessed by the authors will
not be hit, and a hit rate near zero means the guessing was the problem.

## Running it in production

It was being served by `python3 main.py`. Werkzeug prints a warning telling you
not to do that, and the warning is right: one process, no worker supervision,
no request limits, and no way to recover a wedged worker. It answers one
question correctly and has nothing to say about a class of thirty arriving at
once.

| | Before | Now |
| --- | --- | --- |
| Server | Flask dev server | `gunicorn`, 1 worker × 8 threads, `gthread` |
| Port | hardcoded `8080` | `$PORT`, which is what Render actually supplies |
| Settings | literals in two files | `config.py`, read from the environment |
| Dependencies | unpinned names | pinned to the versions the numbers came from |
| Output | `print(..., flush=True)` | `logging`, with levels |
| Deploy config | `.replit` claiming Cloud Run | `render.yaml` and `Procfile` |

One worker and eight threads, not the reverse. Each worker holds its own copy
of the index, and memory is the scarce resource on a free instance; threads
cost almost nothing because a request spends effectively all of its time
blocked on the API and releases the GIL while it waits.

`preload_app` is off deliberately, and this is the kind of thing that fails
silently. Preloading imports the app in the master and forks workers, which is
normally the right trade. But `warm_start()` does its work on a background
thread, and **a thread does not survive `fork`** — it would start in the master
and vanish from the worker actually serving requests, leaving the warm-up
working only when it happened to win a race. Without preload each worker warms
itself. Boot is marginally slower and correct.

### On the free tier, none of this is your biggest latency problem

Everything above is worth having and none of it is the main event. Render's
free tier spins a service down after 15 minutes idle, and a spun-down service
has to boot a container from cold before it sees the request.

Measured against the live deployment, first request after a long idle versus
the same endpoint warm:

| Live `/ask` | Total |
| --- | --- |
| First request after idle | 12,349 ms |
| Warm | 3,879 to 5,245 ms |
| **Spin-up** | **~7 to 8 s** |

Against that, the 0.8 to 1.3 s `warm_start()` saves is rounding.

An earlier attempt to measure this hit 183 ms and concluded the service was not
spinning down at all. That measurement was wrong: it idled 17 minutes and then
timed `/health`, but something kept the service alive during the window, so it
timed a warm process and reported the result as if it were cold. Worth stating
because the failure mode is not obvious — a cold-start measurement that comes
back fast has not proved the service stays warm, it has failed to catch it
asleep, and those two look identical from outside.

A school chatbot is idle most of the day and then used in bursts, which is
precisely the traffic shape that pays this cost on almost every burst. **The
single highest-impact change available is an external uptime pinger** hitting
`/health` every 10 minutes so the service never spins down. It is free, it is
not code, and it is worth more than every optimisation in this README combined.

Two things to know before doing it. The free tier allows 750 instance-hours a
month and a month is about 730 hours, so one always-on service fits and a
second one does not. And `/health` is the right target because it is cheap —
it reports on the loaded index and makes no API call. `/health?deep=1` spends
an embedding call and is for debugging a broken deploy, not for a pinger.

## Limits

* **Both labeled sets are small**: 20 grounding cases, 25 retrieval questions. A
  perfect F1 on 20 cases means the judge did not fail here, not that it does not
  fail. The injected false claims are also fairly blunt, and subtle unsupported
  claims are the harder untested case.
* **Thresholds are specific** to ada-002 on this corpus and do not transfer to
  another embedding model.
* **Cost figures are computed** from published per-token rates, not read off a
  billing dashboard.
* **Retrieval is still the weak point.** Top-3 source accuracy is 66.7%, though
  that metric now undercounts: a link chunk is attributed to the page it was
  found on, not the page it points at, so a correct answer can score as a miss.
  Link availability, which is what the "where do I find X" case actually needs,
  is 100%.
* **The gate separates perfectly, by 0.003.** Answerable questions bottom out
  at 0.795 and out-of-scope reach 0.792, so 0.793 scores 100%. That margin will
  not survive contact with a larger question set, so the configured cutoff is
  0.79: it leans toward answering, on the grounds that a missed caution still
  faces two more checks while a spurious one teaches readers to ignore the
  panel.

* **The hallucination rates are 50 samples at temperature 1.0.** Reruns move
  every figure by several points — the unsupported-claim rate came out at 24%,
  28% and 24% on three consecutive runs of the same harness. Treat them as a
  range, not a measurement. The zeros are the sturdiest numbers there, since a
  zero over 50 trials still bounds the rate loosely rather than proving it.
* **The out-of-scope set is blunt.** Questions like the capital of France are
  plainly uncovered, and the gate separates them easily. The untested case is
  the near-miss, where the site half answers and the honest reply is partial.

Ranked next steps: **fix the 10.0% over-refusal**, which is now the largest
measured defect and is a prompt problem rather than a retrieval one; hybrid
retrieval (BM25 plus dense, since the remaining misses are lexical); semantic
chunking on headings instead of a fixed 150-word window; a larger grounding set
with subtler hallucinations; and scheduled re-scraping so a year rollover cannot
silently empty a page again.

## Reproducing the numbers

Every figure above comes from a script in `eval/`, with raw output committed in
`eval/results/`.

First, a key. Copy `.env.example` to `.env` and put the same key the deployment
uses in it, so what you measure locally is what students actually get.

A `.env` here **overrides** the machine's environment, which is deliberate.
`OPENAI_API_KEY` is commonly already set globally for some other project, and
the wrong key fails in a way that wastes an afternoon: it authenticates, lists
models happily, and then reports `credit_balance_exhausted` on the first
billable call. That reads like a billing problem and is actually the wrong
account. Keys belong to projects, not machines.

`.env` is gitignored and must stay so. It never reaches Render, where the
dashboard's environment variables are the only source, so there is nothing for
the override to collide with in deployment.

```bash
python eval/corpus_stats.py           # corpus and threshold analysis, no API key
python eval/link_grounding_eval.py    # link check precision and recall, no API key
python eval/retrieval_eval.py         # retrieval accuracy and gate calibration
python eval/detection_ablation.py     # all five detector architectures
python eval/latency.py                # cold start, cache, per-stage timings
python eval/hallucination_rate.py     # end-to-end hallucination rate, 50 trials
python prewarm.py                     # build the pre-verified answer cache

VRHS_REPEATS=3 python eval/hallucination_rate.py   # more samples per question

# point the corpus evals at either index
VRHS_EMBEDDINGS=data/vrhs_embeddings_baseline.json python eval/retrieval_eval.py
```

## Layout

| Path | Purpose |
| --- | --- |
| `main.py` | Flask app: scraping, embedding, retrieval, streaming `/ask` |
| `config.py` | Every setting, read from the environment |
| `hallucination.py` | The three checks and the note they produce |
| `corpus.py` | Boilerplate stripping and dedupe, run before embedding |
| `gdocs.py` | Reads linked Google Docs, and Drive filenames for dating |
| `prewarm.py` | Generates and verifies answers ahead of time |
| `gunicorn.conf.py` | Production server config, and why `preload_app` is off |
| `render.yaml`, `Procfile` | Deployment |
| `eval/detectors.py` | All five detection architectures, including the unshipped ones |
| `eval/hallucination_rate.py` | End-to-end rate through the real answer path |
| `eval/` | Harness, labeled fixtures, committed raw results |
| `data/vrhs_embeddings.json` | Production index, 218 chunks |
| `data/vrhs_embeddings_baseline.json` | Same scrape without hygiene, 263 chunks |
