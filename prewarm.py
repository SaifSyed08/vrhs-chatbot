"""
Generate and verify answers ahead of time, so common questions cost nothing.

The two API calls in front of a live answer have a floor. Embedding the
question is 208 ms and gpt-4.1's first token is 451 ms, both round trips whose
cost is on OpenAI's side of a Cloudflare edge. A question nobody has asked
before cannot realistically be answered in under about 700 ms, and no amount
of tuning changes that.

The way under the floor is to have answered already. This script writes
data/answer_cache.json, which main.py serves from memory without calling
anything.

Two things make it more than a speed trick.

**Verification stops being advisory.** Everywhere else in this codebase the
checks annotate an answer the reader has already seen, because an answer is
only checkable once it is whole and by then the tokens are on screen. Here
there is no reader yet, so a variant that fails a check is discarded instead
of shipped with a warning. A cached answer is not merely faster than a live
one, it has passed a bar a live one cannot be held to.

**Variants vary wording, not facts.** Several answers are generated per
question and rotated, so the bot does not repeat one phrasing verbatim all
year. That is only safe if the variants agree, and "they all passed the
grounding check" does not establish that - two answers can each be supported
by the context and still tell a student different things. So a set is admitted
only when every variant cites the same pages. Divergent link sets mean the
answers differ in what they actually direct someone to do, and the whole set
is rejected rather than gambled on.

Questions come from the eval fixtures and, when it exists, from
data/feedback.json - the questions students actually asked. The second source
is the better one, and it is the reason /health reports the cache hit rate:
a list guessed by the authors will not be hit.

    OPENAI_API_KEY=... python prewarm.py
    VRHS_VARIANTS=5 python prewarm.py
"""

import datetime
import json
import os
import sys

os.environ.setdefault("VRHS_NO_WARM", "1")

import config  # noqa: E402
import hallucination  # noqa: E402
import main  # noqa: E402

VARIANTS = int(os.getenv("VRHS_VARIANTS", "3"))
FIXTURES = os.path.join("eval", "fixtures", "questions.json")
FEEDBACK = os.path.join("data", "feedback.json")
OUT = os.getenv("VRHS_ANSWER_CACHE", os.path.join("data", "answer_cache.json"))


def questions():
    """Questions worth pre-answering, best source first, de-duplicated."""
    seen, out = set(), []

    def add(q, origin):
        key = main.cache_key(q)
        if key and key not in seen:
            seen.add(key)
            out.append((q, origin))

    # Real questions first. These are what students actually typed.
    try:
        with open(FEEDBACK, encoding="utf-8") as f:
            for entry in json.load(f):
                q = (entry.get("question") or "").strip()
                if q:
                    add(q, "feedback")
    except (OSError, json.JSONDecodeError):
        pass

    # Only the answerable fixtures. Pre-generating a refusal to an out-of-scope
    # question saves nothing worth having and freezes a wording that the live
    # path produces perfectly well.
    try:
        with open(FIXTURES, encoding="utf-8") as f:
            for case in json.load(f).get("answerable", []):
                add(case["question"], "fixture")
    except (OSError, json.JSONDecodeError):
        pass

    return out


def generate(question):
    """One candidate answer, through the same path the route uses."""
    context, stats, sources = main.get_relevant_context(question)

    response = main.client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[{
            "role": "system",
            "content": main.dated_prompt()
        }, {
            "role": "user",
            "content": "Context:\n" + context + "\n\nQuestion: " + question
        }])
    answer = (response.choices[0].message.content or "").strip()

    verdict = hallucination.verify_answer(main.client, answer, context, stats,
                                          question)
    return answer, sources, verdict


def admit(question):
    """Generate VARIANTS answers and return the ones fit to serve.

    Returns (variants, reason). An empty list means this question is left to
    the live path, which is a perfectly good outcome - it is the same answer
    the bot gives today, just not pre-paid.
    """
    kept = []
    for _ in range(VARIANTS):
        answer, sources, verdict = generate(question)

        # The bar is higher than the live path's, deliberately. Live, any of
        # these produces an answer with a caution attached. Here it produces
        # nothing, because there is a live path to fall back on and no reason
        # to freeze a flawed answer into a file and serve it all year.
        if verdict.bad_links:
            continue
        if verdict.unsupported:
            continue
        if verdict.retrieval_level != "solid":
            continue
        if len(answer) < 40:
            continue

        kept.append({
            "answer": answer,
            "sources": sources,
            "notice": [],
            "retrieval": {
                "level": verdict.retrieval_level,
                "top": verdict.retrieval_top,
            },
            "links": sorted(set(hallucination.cited_urls(answer))),
        })

    if not kept:
        return [], "no variant passed"

    # Same pages, or none of them. Two answers can each be grounded and still
    # send a student to different places, and rotating between those is worse
    # than never having cached anything.
    linksets = {tuple(v["links"]) for v in kept}
    if len(linksets) > 1:
        return [], "variants cited different pages (%d sets)" % len(linksets)

    for v in kept:
        del v["links"]
    return kept, "ok"


def main_run():
    qs = questions()
    if not qs:
        sys.exit("no questions found - need eval fixtures or data/feedback.json")

    docs, _ = main.load_index()
    if docs is None:
        sys.exit("no index at " + config.EMBEDDINGS_PATH)

    print("%d questions, %d variants each, model %s"
          % (len(qs), VARIANTS, config.CHAT_MODEL))
    print()

    entries, admitted, rejected = {}, 0, 0
    for question, origin in qs:
        variants, reason = admit(question)
        if variants:
            entries[main.cache_key(question)] = {
                "question": question,
                "origin": origin,
                "variants": variants,
            }
            admitted += 1
            print("  keep    %-52s %d/%d variants"
                  % (question[:52], len(variants), VARIANTS))
        else:
            rejected += 1
            print("  skip    %-52s %s" % (question[:52], reason))

    blob = {
        "index_fingerprint": main.index_fingerprint(docs),
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "model": config.CHAT_MODEL,
        "variants_requested": VARIANTS,
        "entries": entries,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2)

    total = sum(len(e["variants"]) for e in entries.values())
    print()
    print("admitted %d questions (%d variants), left %d to the live path"
          % (admitted, total, rejected))
    print("index fingerprint %s" % blob["index_fingerprint"])
    print("written to %s" % OUT)
    print()
    print("A rebuilt index retires this file automatically: main.py compares")
    print("the fingerprint and ignores a cache that no longer matches, so a")
    print("re-scrape cannot leave stale answers being served. Re-run this")
    print("after any rebuild.")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    main_run()
