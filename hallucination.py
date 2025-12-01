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

# Cosine similarity floor for text-embedding-ada-002. Unrelated English text
# still lands around 0.70 with this model, so the useful floor sits well above
# zero. Below WEAK the answer is unsupported; between WEAK and SOLID it is thin.
SIMILARITY_SOLID = 0.82
SIMILARITY_WEAK = 0.76

# Model used for the grounding pass. Deliberately not the answering model - a
# grader that shares the generator's mistakes will happily ratify them.
GRADER_MODEL = "gpt-4o-mini"

MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^)\s]+)\)")


class Verdict:
    """Result of running the checks over one answer."""

    def __init__(self):
        self.retrieval_score = None
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


def score_retrieval(scores):
    """Grade how well the knowledge base actually matched the question."""
    if not scores:
        return None, "none"

    top = max(scores)
    if top >= SIMILARITY_SOLID:
        return top, "solid"
    if top >= SIMILARITY_WEAK:
        return top, "weak"
    return top, "none"


def find_ungrounded_links(answer, context):
    """Return URLs cited in the answer that do not appear in the context."""
    bad = []
    for _label, url in MARKDOWN_LINK.findall(answer):
        # Trailing punctuation sneaks into the capture on occasion.
        clean = url.rstrip(".,;:")
        if clean not in context:
            bad.append(clean)
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


def verify_answer(client, answer, context, scores):
    """Run every check over one answer and return a Verdict."""
    verdict = Verdict()
    verdict.retrieval_score, verdict.retrieval_level = score_retrieval(scores)
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
