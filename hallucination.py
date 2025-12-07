"""
Hallucination detection layer.

The chatbot answers from a small RAG index scraped off the VRHS site, so the
realistic failure mode is not wild invention - it is a confident answer built
on context that does not actually say what the answer claims. Three checks run
against every response:

  1. Retrieval confidence - if the closest chunk in the knowledge base is not
     close enough to the question, the model is answering from pretraining
     rather than from VRHS pages.
  2. Link grounding - every [label](url) in the answer must appear verbatim in
     the retrieved context. Invented school URLs are the failure that actually
     costs a student time, and they are cheap to catch exactly.
  3. Claim grounding - a second, cheaper model pass that sees only the context
     and the answer and reports which statements the context does not support.

The checks are advisory. They never suppress an answer; they attach a notice to
it so the reader knows which parts to verify.
"""

import json
import re

# Retrieval gating thresholds, calibrated in eval/retrieval_eval.py over 25
# real questions (15 the corpus covers, 10 it does not).
#
# This setting was wrong twice before it was measured, which is worth recording:
#
#   1. First guess was an absolute cutoff picked by intuition (0.82 / 0.76).
#   2. Corpus analysis appeared to kill absolute cutoffs outright: the median
#      cosine between two unrelated chunks is 0.8817, and the 5th percentile of
#      a perfect chunk-to-chunk match is 0.8798, so the two overlap. That
#      argued for a scale-free standard-score margin instead.
#   3. Measuring against real questions reversed it again. Step 2 used chunks
#      as stand-in queries, and chunks are the one thing that carries the site's
#      navigation boilerplate - which is what inflated the baseline to 0.88.
#      A real question is short and carries none of it, so query-to-chunk
#      similarity lands far lower and separates cleanly:
#
#        answerable questions    top-1 cosine >= 0.782
#        out-of-scope questions  top-1 cosine <= 0.773
#
#      Head to head on that question set, raw cosine separated the two at 100%
#      accuracy; the standard-score margin managed 76%.
#
# So: absolute cutoff, with the ambiguous band between the two measured
# populations treated as weak. The margin is still computed and logged as a
# diagnostic. Caveat: 25 questions is a small calibration set, and these
# thresholds are specific to text-embedding-ada-002 on this corpus.
SIMILARITY_SOLID = 0.78
SIMILARITY_WEAK = 0.77

# Model used for the grounding pass. Deliberately not the answering model - a
# grader that shares the generator's mistakes will happily ratify them.
GRADER_MODEL = "gpt-4o-mini"

MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")
# Bare URLs count too. Measured on eval/fixtures/link_cases.json: checking
# only markdown links caught 9 of 11 fabricated URLs (recall 0.818) - the
# model drops a raw link into prose often enough to matter.
BARE_URL = re.compile(r"https?://[^\s<>()\[\]]+")


class Verdict:
    """Result of running the checks over one answer."""

    def __init__(self):
        self.retrieval_z = None
        self.retrieval_top = None
        self.retrieval_level = "unknown"
        self.bad_links = []
        self.unsupported = []
        self.grader_failed = False

    @property
    def grounded(self):
        return (self.retrieval_level == "solid" and not self.bad_links
                and not self.unsupported)

    def notice(self):
        """Markdown appended to the streamed answer, or '' if all checks pass."""
        if self.grounded:
            return ""

        lines = []

        if self.retrieval_level == "none":
            lines.append(
                "I could not find anything on the VRHS site that covers this, "
                "so the answer above is not based on school pages.")
        elif self.retrieval_level == "weak":
            lines.append(
                "The closest match in the school pages was only loosely related "
                "to this question, so please double-check the answer above.")

        if self.bad_links:
            shown = ", ".join(self.bad_links[:3])
            lines.append(
                "These links were not in my sources and may not exist: " + shown)

        if self.unsupported:
            for claim in self.unsupported[:3]:
                lines.append("The school pages I read do not confirm: " + claim)

        if self.grader_failed:
            lines.append("My grounding check did not finish, so this answer is "
                         "unverified.")

        if not lines:
            return ""

        body = "\n".join("- " + line for line in lines)
        return ("\n\n---\n**Heads up - please verify this one.**\n" + body +
                "\nThe [staff directory](https://vrhs.leanderisd.org/directory) "
                "is the reliable source if you need to be sure.")


def score_retrieval(stats):
    """Grade whether the knowledge base actually covers this question."""
    if not stats or stats.get("top") is None:
        return None, "none"

    top = stats["top"]
    if top >= SIMILARITY_SOLID:
        return top, "solid"
    if top >= SIMILARITY_WEAK:
        return top, "weak"
    return top, "none"


def cited_urls(answer):
    """Every URL the answer points at, markdown-wrapped or bare."""
    urls, spans = [], []

    for match in MARKDOWN_LINK.finditer(answer):
        urls.append(match.group(2))
        spans.append(match.span())

    for match in BARE_URL.finditer(answer):
        # Skip anything already counted inside a markdown link.
        if any(start <= match.start() < end for start, end in spans):
            continue
        urls.append(match.group(0))

    # Trailing sentence punctuation sneaks into the capture on occasion.
    return [u.rstrip(".,;:") for u in urls]


def find_ungrounded_links(answer, context):
    """Return URLs cited in the answer that do not appear in the context."""
    seen, bad = set(), []
    for url in cited_urls(answer):
        if url not in context and url not in seen:
            seen.add(url)
            bad.append(url)
    return bad


GRADER_PROMPT = """You are checking whether an answer is supported by source text.

You will be given SOURCE (text scraped from a high school's website) and ANSWER.
List every factual claim in ANSWER that SOURCE does not support. A claim counts
as unsupported if SOURCE does not state it, even if you personally believe it is
true - you are grading against SOURCE only, never against your own knowledge.

Ignore greetings, offers to help, closing pleasantries, and hedged suggestions
to contact the school. Those are not factual claims.

Reply with JSON only, in this shape:
{"unsupported": ["claim one", "claim two"]}

Use an empty list when every claim is supported."""


def check_claim_grounding(client, answer, context):
    """Ask a second model which claims the context fails to support."""
    response = client.chat.completions.create(
        model=GRADER_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{
            "role": "system",
            "content": GRADER_PROMPT
        }, {
            "role": "user",
            "content": "SOURCE:\n" + context + "\n\nANSWER:\n" + answer
        }])

    parsed = json.loads(response.choices[0].message.content)
    claims = parsed.get("unsupported", [])

    # Guard against the grader returning a bare string or nested junk.
    return [str(c).strip() for c in claims if str(c).strip()][:5]


def verify_answer(client, answer, context, stats):
    """Run every check over one answer and return a Verdict."""
    verdict = Verdict()
    verdict.retrieval_top, verdict.retrieval_level = score_retrieval(stats)
    verdict.retrieval_z = (stats or {}).get("z")  # diagnostic only
    verdict.bad_links = find_ungrounded_links(answer, context)

    # An empty or trivially short answer has nothing worth grading.
    if len(answer.strip()) < 40:
        return verdict

    try:
        verdict.unsupported = check_claim_grounding(client, answer, context)
    except Exception as e:
        print("Grounding check failed: " + str(e))
        verdict.grader_failed = True

    return verdict
