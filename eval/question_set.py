"""
Run every question the interface offers, and judge what comes back.

The landing composer suggests 39 questions and cycles them through the
placeholder on hover. Every one of them is a promise: the interface put it in
front of a reader, so the bot had better be able to answer it. Nothing until
now checked that. They were written by reading what ingest found, which is a
reasonable way to guess and not a way to know - and the corpus has been
rebuilt three times since.

**No answer key.** The obvious way to build this would be to write down the
right answer for each question and compare. That fails for the reason the
whole project exists: the answers change. A bell time moves, a club changes
room, the graduation date rolls over, and a fixture that was right in March
quietly starts failing the bot for being correct. Worse, an answer key is only
as good as the afternoon somebody spent writing it, and it encodes that
afternoon's mistakes forever.

So this grades against evidence instead, and the evidence is gathered fresh
each run:

  * The answer is checked against the context the bot actually retrieved -
    is every specific claim in it supported by text the bot was given?
  * Whether the bot answered at all, or deflected to a link, is classified by
    a model rather than by looking for phrases like "I don't have".
  * When the bot fails to answer, a second pass goes and looks: it reads the
    pages the corpus links and searches the district site, and reports whether
    the information was reachable. That is the difference between "the site
    does not say" and "the bot could not find what the site says", and only
    the second is a bug in this repository.

Questions come from templates/index.html rather than from a copy here. One
list, so the eval cannot drift away from what the interface promises.

Both the generator and the two judges run at temperature 0. Production does
not, and should not - varied phrasing is worth having there. But an eval that
inherits it cannot tell a change from a coin flip, and two questions looked
like regressions on a retrieval change before this was pinned, then turned out
to produce identical answers at both settings.

    python eval/question_set.py            # everything
    python eval/question_set.py --limit 8  # a quick pass
"""

import argparse
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("VRHS_NO_WARM", "1")

import config          # noqa: E402
import hallucination   # noqa: E402
import livesearch      # noqa: E402
import main as app     # noqa: E402

TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "templates", "index.html")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

JUDGE_MODEL = os.getenv("VRHS_JUDGE_MODEL", "gpt-4.1")


def questions():
    """The suggestion list, read out of the template that serves it."""
    markup = io.open(TEMPLATE, encoding="utf-8").read()
    block = markup.split("const QUESTIONS = [", 1)[1].split("];", 1)[0]
    return re.findall(r'"([^"]+)"', block)


JUDGE_PROMPT = """You are grading one answer from a school chatbot.

You are given the question, the answer, and the exact context the bot
retrieved to write it. Judge only what is in front of you. Do not use outside
knowledge about the school, and do not reward or punish an answer for matching
what you personally expect.

Report three things.

"outcome" - one of:
  "answered"   the answer states the information the question asked for
  "deflected"  the answer does not state it, and instead points at a page,
               a document, a directory or the front office
  "refused"    the answer says it does not have the information
  "clarifying" the answer asks the reader what they mean

Deciding between "answered" and "deflected" turns on what was asked, not on
whether a link appears.

Some questions ask WHERE something is - "where is the staff directory", "where
is the library website", "how do I log into X". The thing being asked for is a
location, so a correct link IS the answer and the outcome is "answered".
Marking those deflected punishes the bot for answering the question.

Other questions ask for a FACT or a PROCEDURE - a time, a date, a room, a
name, a list, the steps to do something. For those, a reply that gives only a
link, a page name or a phone number is "deflected" however helpfully it is
phrased, because the reader still has to go and find the answer themselves.

A reply that both states the information and links a page is always
"answered".

"supported" - true only if every specific claim in the answer appears in the
context. Specific means a time, a date, a room, a name, an email, a fee, a
deadline or a requirement. General framing is not a claim.

Ignore URLs entirely. The bot has a separate check that verifies every link
against the context character by character, and it is stricter than you;
grading them twice only produces disagreements.

"unsupported_claims" - the specific claims that do not appear in the context,
verbatim from the answer, as a list of strings. Empty when supported is true.

Reply with JSON only:
{"outcome": "...", "supported": true, "unsupported_claims": []}"""


def judge(client, question, answer, context):
    body = ("Question:\n%s\n\nAnswer:\n%s\n\nContext the bot retrieved:\n%s"
            % (question, answer, context[:14000]))
    response = app.with_retry(lambda: client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": JUDGE_PROMPT},
                  {"role": "user", "content": body}],
        temperature=0,
        response_format={"type": "json_object"}), attempts=5)
    return json.loads(response.choices[0].message.content)


REACH_PROMPT = """You are checking whether a school website actually contains
the answer to a question, so that a chatbot's failure to answer it can be
attributed correctly.

You are given the question and text gathered from the school's own pages, the
documents they link, and the district site. Say whether the answer to the
question is present in that text.

"present" - true if the text states the answer, even partially.
"evidence" - the sentence or two that states it, verbatim. Empty if absent.

Reply with JSON only: {"present": true, "evidence": "..."}"""


def reachable(client, question, docs, matrix):
    """Whether the answer exists anywhere the bot could have looked.

    Deliberately looser than retrieval: twelve chunks instead of five, no
    quotas, no gate, plus whatever the district search turns up. The point is
    not to answer the question but to settle who is at fault. If the answer is
    in here and the bot did not give it, the retrieval is wrong. If it is not
    in here, the site does not say and the suggestion should not have been
    offered.
    """
    import numpy as np

    vector = np.array(app.embed_query(question), dtype=np.float64)
    vector /= np.linalg.norm(vector)
    sims = matrix @ vector
    best = np.argsort(sims)[::-1][:12]
    gathered = [docs[i]["text"] for i in best]

    try:
        for passage in livesearch.lookup(question, app.embed_texts):
            gathered.append("From %s: %s" % (passage["label"],
                                             passage["text"]))
    except Exception:
        pass

    body = ("Question:\n%s\n\nText gathered from the school and district "
            "sites:\n%s" % (question, "\n\n".join(gathered)[:16000]))
    response = app.with_retry(lambda: client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": REACH_PROMPT},
                  {"role": "user", "content": body}],
        temperature=0,
        response_format={"type": "json_object"}), attempts=5)
    return json.loads(response.choices[0].message.content)


def answer_for(question):
    """One answer, through the same path /ask uses."""
    context, stats, sources = app.get_relevant_context(
        app.retrieval_query(question, []))

    live = []
    if stats.get("top_prose", 1.0) < config.LIVE_TRIGGER:
        try:
            live = livesearch.lookup(question, app.embed_texts)
        except Exception:
            live = []

    live_leads = bool(live) and stats.get("top", 1.0) < \
        hallucination.SIMILARITY_WEAK
    if live:
        gap = "\n\n"
        context = context + gap + gap.join(
            "From %s (%s): %s" % (c["label"], c["url"], c["text"])
            for c in live)
        found = [{"url": c["url"], "label": c["label"]} for c in live]
        sources = found if live_leads else sources + found

    messages = [{"role": "system", "content": app.dated_prompt()},
                {"role": "user",
                 "content": app.context_block(context, stats, live, live_leads)
                            + "\n\nQuestion: " + question}]
    # Pinned, unlike production. The bot answers at the default temperature
    # because varied phrasing is worth having; an eval that does the same
    # cannot tell a change from a coin flip. Two questions looked like
    # regressions when PROSE_SLOTS went from three to five and turned out to
    # produce byte-identical answers at both settings once this was set.
    reply = app.with_retry(lambda: app.client.chat.completions.create(
        model=config.CHAT_MODEL, temperature=0, messages=messages),
        attempts=5)
    return (reply.choices[0].message.content.strip(), context, stats,
            sources, bool(live))


def main_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(OUT_DIR,
                                                      "question_set.json"))
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")

    qs = questions()
    if args.limit:
        qs = qs[:args.limit]
    print("%d suggested questions, model %s, judge %s\n"
          % (len(qs), config.CHAT_MODEL, JUDGE_MODEL))

    docs, matrix = app.load_index()
    if docs is None:
        sys.exit("no index")

    rows = []
    for n, question in enumerate(qs, 1):
        started = time.time()
        try:
            answer, context, stats, sources, live = answer_for(question)
            verdict = judge(app.client, question, answer, context)
        except Exception as e:
            print("%2d. ERROR  %s  (%s)" % (n, question[:52], e))
            rows.append({"question": question, "error": str(e)})
            continue

        row = {
            "question": question,
            "answer": answer,
            "outcome": verdict.get("outcome"),
            "supported": bool(verdict.get("supported")),
            "unsupported_claims": verdict.get("unsupported_claims") or [],
            "top": round(stats.get("top", 0), 4),
            "top_prose": round(stats.get("top_prose", 0), 4),
            "used_live": live,
            "sources": [s.get("label") for s in sources],
            "seconds": round(time.time() - started, 1),
        }

        # Only failures get the second pass, because it is the expensive one
        # and there is nothing to attribute when the bot answered.
        if row["outcome"] in ("deflected", "refused"):
            try:
                found = reachable(app.client, question, docs, matrix)
                row["answer_reachable"] = bool(found.get("present"))
                row["evidence"] = (found.get("evidence") or "")[:400]
            except Exception as e:
                row["answer_reachable"] = None
                row["evidence"] = "check failed: %s" % e

        rows.append(row)

        flag = {"answered": "ok  ", "deflected": "DEFL", "refused": "REFU",
                "clarifying": "CLAR"}.get(row["outcome"], "????")
        note = ""
        if row.get("answer_reachable") is True:
            note = "  <- but the site does say"
        elif row.get("answer_reachable") is False:
            note = "  (site really does not say)"
        if not row["supported"]:
            note += "  UNSUPPORTED x%d" % len(row["unsupported_claims"])
        print("%2d. %s %-52s %.3f%s" % (n, flag, question[:52],
                                        row["top_prose"], note))

    graded = [r for r in rows if "outcome" in r]
    counts = {}
    for r in graded:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1

    bad = [r for r in graded if r["outcome"] in ("deflected", "refused")]
    fixable = [r for r in bad if r.get("answer_reachable")]
    unsupported = [r for r in graded if not r["supported"]]

    print("\n" + "=" * 68)
    total = len(graded) or 1
    for name in ("answered", "deflected", "refused", "clarifying"):
        got = counts.get(name, 0)
        print("  %-11s %2d  %4.0f%%" % (name, got, 100.0 * got / total))
    print("  %-11s %2d  %4.0f%%   (a claim not in the retrieved context)"
          % ("unsupported", len(unsupported),
             100.0 * len(unsupported) / total))
    print("  %-11s %2d          of %d failures were reachable anyway"
          % ("recoverable", len(fixable), len(bad)))

    os.makedirs(OUT_DIR, exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": config.CHAT_MODEL, "judge": JUDGE_MODEL,
                   "chunks": len(docs), "rows": rows}, f, indent=2)
    print("\nwritten to %s" % args.out)


if __name__ == "__main__":
    main_run()
