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

import config

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
SIMILARITY_SOLID = config.SIMILARITY_SOLID
SIMILARITY_WEAK = config.SIMILARITY_WEAK

# Recalibrated after the corpus grew to a full crawl. The populations now
# separate completely on the question set: answerable questions bottom out at
# 0.795 and out-of-scope questions top out at 0.792, so a cutoff of 0.793
# scores 100%. That is not the number used. A 0.003 margin measured on 25
# questions will not hold, and the two errors are not symmetric: a false
# "solid" only means no caution panel on an answer the other two checks still
# inspect, while a false "weak" puts a warning on a good answer and teaches
# readers to ignore the panel. 0.79 leans toward answering on purpose.

# Model used for the grounding pass. Deliberately not the answering model - a
# grader that shares the generator's mistakes will happily ratify them.
GRADER_MODEL = config.GRADER_MODEL

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
        # Set only when the reply gave up rather than answered. Suppresses the
        # retrieval-level caution, which exists to flag misplaced confidence
        # and has nothing to say about an answer that claimed nothing.
        self.declined = False
        # Whether the reader will be shown any sources at all. The
        # unsupported-claim notice tells them to check the sources below, and
        # there is nothing below when retrieval came up empty.
        self.has_sources = True

    @property
    def grounded(self):
        if self.bad_links:
            return False
        if len(self.unsupported) >= 2 and self.has_sources:
            return False
        # A reply that declined is treated as clean whatever retrieval scored.
        # Nothing was asserted, so there is nothing for a reader to verify.
        return self.retrieval_level == "solid" or self.declined

    def notice_points(self):
        """At most two short points, or [] when every check passed.

        An earlier version listed every unsupported claim it found, which meant
        a weak answer could be buried under five bullets of hedging. The panel
        is a caution, not a report: it names the most actionable problem and
        stops. Order matters, a fabricated link is worth more to a reader than
        a general warning.

        Kept short and impersonal. An earlier pass wrote these as sentences in
        the first person - "I did not find a close match on the school's pages,
        so this may not be right" - which reads as the bot apologising and is
        three lines of text under an answer somebody is trying to skim. A
        caution should be readable at a glance or it is not a caution.
        """
        if self.grounded:
            return []

        points = []

        if self.bad_links:
            first = self.bad_links[0]
            extra = len(self.bad_links) - 1
            tail = f" and {extra} other" + ("s" if extra > 1 else "") if extra else ""
            points.append(f"Link not found on the school site: "
                          f"{first}{tail}")

        # At most one point about grounding, and the retrieval one wins.
        #
        # These used to stack: "No close match on the school's pages" followed
        # by "Some details may not align with the sources below" said the same
        # worry twice, and the second is implied by the first - if retrieval
        # found nothing close, of course the details may not line up. One line
        # gets read; two get skipped.
        if not self.declined and self.retrieval_level == "none":
            points.append("Your query had no close match on the school's "
                          "pages, so this may not be right.")
        elif not self.declined and self.retrieval_level == "weak":
            points.append("Your query had only a loose match on the school's "
                          "pages.")
        elif self.unsupported and self.has_sources:
            # Two or more, not one.
            #
            # A single flagged claim in a long answer is usually the grader
            # being marginal on a sentence the source states in other words -
            # the failure mode measured all through this file. Two or more is a
            # pattern rather than a wobble, and the panel is worth spending on
            # a pattern. The claims are still recorded either way; this decides
            # only whether the reader is interrupted.
            if len(self.unsupported) >= 2:
                points.append("Some details may not align with the sources "
                              "below.")
        elif self.grader_failed:
            points.append("This answer wasn't fully checked.")

        return points[:2]


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


# URLs the system prompt itself hands the model as a fallback when the corpus
# cannot answer. They are verified site pages, but they do not arrive through
# retrieval, so without this the check flags the assistant for following its
# own instructions.
ALWAYS_GROUNDED = {
    "https://vrhs.leanderisd.org/directory",
}


def find_ungrounded_links(answer, context):
    """Return URLs cited in the answer that do not appear in the context."""
    seen, bad = set(), []
    for url in cited_urls(answer):
        if url in ALWAYS_GROUNDED:
            continue
        if url not in context and url not in seen:
            seen.add(url)
            bad.append(url)
    return bad


# The exclusion list below is long and specific because a short one did not
# work. It used to read "ignore greetings, offers to help, closing
# pleasantries, and hedged suggestions to contact the school", and the grader
# ignored none of them.
#
# Measured in eval/hallucination_rate.py: on out-of-scope questions, where the
# bot correctly refuses, 18 of 20 refusals were flagged as containing an
# unsupported claim. The flagged text was the refusal itself.
#
#   "I don't have the information about the plot of Hamlet."
#   "I'm here to help with inquiries related to Vista Ridge High School."
#   "contact the front office for assistance."
#
# None of those are assertions about the school, so no source could ever
# support them, and the grader is right that the context does not state them.
# It was asked the wrong question. A sentence describing what the assistant
# knows is not a claim about the world, and grading it against a scrape of the
# school website is a category error.
#
# The cost was not academic. Every one of those refusals shipped with a
# "Some details are not confirmed by the pages I read" caution attached to it,
# so the panel fired hardest on the answers behaving best. That is the failure
# the whole layer is supposed to avoid: a warning a reader learns to ignore is
# worse than no warning, because it spends the credibility the real ones need.
#
# Naming the exclusions concretely rather than by category fixed it, and the
# distinction the prompt now leads with - claims about the school, not claims
# about the assistant - is what carries the weight.
# Two calls, because the two questions need different framing and asking one
# model to answer both made it worse at each.
#
# DETECTOR_PROMPT below is the original, and it is kept verbatim because it
# measures precision 1.00 and recall 1.00 on the 20 labelled cases. Its problem
# was never detection. It was that in production it also reported refusals,
# referrals and sign-offs as unsupported claims - 54 of 62 flagged items in an
# audit of eval/results/hallucination_rate_trials.json were of that kind.
#
# Three attempts to fix that by instruction failed. Naming the exclusions,
# naming them again with examples, then adding a verbatim-quoting requirement
# and worked cases, moved the flag rate 38.7% to 34.7%; the grader kept
# reporting the same sentences and merely quoted them more accurately.
#
# Restructuring the task as "label every sentence, then report only the ones
# you called a fact" fixed precision outright, 38.7% to 5.3%. It also dropped
# recall from 1.000 to 0.556, which is a far worse trade: a missed
# hallucination reaches a reader silently, an over-warning only annoys one.
# Two things broke. "Completed forms should be emailed directly to the
# principal, Dr. Keith Morgan" was labelled a referral, because it is one in
# form while fabricating a name and a procedure in substance. And having
# sorted sentences into kinds first, the model became noticeably more willing
# to call the remaining ones supported.
#
# So the jobs are separated rather than merged. The detector runs unchanged and
# stays suspicious, and a second pass decides which of the things it found were
# claims at all. The filter only runs when the detector found something, so the
# common case is still one call, and both are post-stream where latency does
# not reach the reader.
# Filter. Sees only what the detector flagged, and decides what kind of
# sentence each one is. Deliberately knows nothing about SOURCE: whether a
# sentence is a claim is a question about the sentence, and giving this pass
# the source text would invite it to re-litigate support and reintroduce the
# laziness that cost recall.
CLAIM_FILTER_PROMPT = """Each ITEM below is a sentence taken from a chatbot
answer for a high school. Label what kind of sentence each one is.

  "school_fact" - asserts something checkable about the school: a time, date,
                  place, cost, person's name, requirement, or procedure.
                  A sentence that directs the reader somewhere is STILL a
                  school_fact when it carries specific detail - a named person,
                  an address, an email, a room, a deadline.
  "no_info"     - says the assistant lacks information or that its source does
                  not cover something.
  "referral"    - sends the reader onward with no specific detail: "contact the
                  front office", "check the staff directory".
  "social"      - greeting, offer of further help, sign-off.

When a sentence could be a referral or a school_fact, choose school_fact. The
cost of the two mistakes is not equal: a fabricated name or deadline waved
through as a referral reaches a reader unchallenged.

Reply with JSON only, same order as the input:
{"labels": ["school_fact", "referral", ...]}

Examples:
  "Please contact the front office."                          -> referral
  "Email the form to principal Dr. Keith Morgan."             -> school_fact
  "I don't have the graduation checklist."                    -> no_info
  "Lunch begins at 12:30 PM."                                 -> school_fact
  "Have a great day!"                                         -> social
  "You can find bell schedules on the calendar page."         -> school_fact"""



DETECTOR_PROMPT = """You are checking whether an answer is supported by source text.

You will be given SOURCE (text scraped from a high school's website) and ANSWER.
List every factual claim in ANSWER that SOURCE does not support. A claim counts
as unsupported if SOURCE does not state it, even if you personally believe it is
true - you are grading against SOURCE only, never against your own knowledge.

A claim is something ANSWER asserts. It is not a topic ANSWER mentions, and it
is not something ANSWER says it could not find. Every item you list must be a
sentence or clause copied WORD FOR WORD from ANSWER. If you cannot copy it
verbatim, it is not a claim and does not belong in the list.

Grade ONLY assertions about the school itself: its schedules, policies, dates,
fees, staff, rooms, events, requirements, or where something can be found.

Do NOT list any of the following. They are not claims about the school, and no
source text could support them:

* Statements about what the assistant knows, has, or was given - for example
  "I don't have that information", "the context doesn't say", "I couldn't find
  the 2026-2027 calendar", "that isn't in the pages I read".
* Statements about the assistant's role or scope - for example "I can only
  help with Vista Ridge High School questions", "that's outside what I cover".
* Suggestions to go somewhere or ask someone - for example "contact the front
  office", "check the staff directory", "you might find it on that page".
  Suggesting a place is not asserting a fact about it.
* Greetings, offers of further help, and closing pleasantries - for example
  "feel free to ask", "have a great day", "hope that helps".
* Statements naming which school year something refers to, when the source
  shows that year - saying a calendar is the 2025-2026 one is reporting the
  source, not adding to it.

A hedged statement is still a claim if it asserts something about the school:
"I believe lunch is at 12:30" is a claim about lunch and must be graded.

Here are real answers and the correct output for each.

ANSWER: "I don't have the specific requirements seniors need before
graduation. I recommend checking with the front office. Have a great day!"
CORRECT: {"unsupported": []}
Why: it asserts nothing about the school. "Senior checklists" and "graduation
requirements" are topics it says it lacks, not claims it makes. Listing them
is the most common way to get this wrong.

ANSWER: "The bell schedules for 2026-2027 are available at the following
links: [Regular Day](https://example.com/a). Let me know if you need more!"
CORRECT: {"unsupported": []}
Why: naming the year the source shows, pointing at a link, and offering help
are not factual assertions to check.

ANSWER: "On early release days (12/19/25 and 5/29/26), the check-out cutoff
is 11:45 AM. Parents must complete the Club Permission Form."
CORRECT: {"unsupported": ["On early release days (12/19/25 and 5/29/26), the
check-out cutoff is 11:45 AM.", "Parents must complete the Club Permission
Form."]}
Why: both state a specific fact about the school - a time and a requirement -
and both can be checked against SOURCE. List them only if SOURCE does not
support them.

Reply with JSON only, in this shape:
{"unsupported": ["exact sentence from ANSWER", "another exact sentence"]}

Use an empty list when every claim is supported. An empty list is the common
and correct answer for a reply that declines, points at a link, or is mostly
pleasantries."""


def detect_unsupported(client, answer, context):
    """Everything the detector thinks the context does not support.

    Kept suspicious on purpose. This pass is measured at recall 1.000 on the 20
    labelled cases and its output is filtered afterwards, so a false positive
    here is cheap and a miss is not recoverable.
    """
    response = client.chat.completions.create(
        model=GRADER_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{
            "role": "system",
            "content": DETECTOR_PROMPT
        }, {
            "role": "user",
            "content": "SOURCE:\n" + context + "\n\nANSWER:\n" + answer
        }])

    parsed = json.loads(response.choices[0].message.content)
    claims = parsed.get("unsupported", [])
    return [str(c).strip() for c in claims if str(c).strip()][:8]


def keep_real_claims(client, claims):
    """Drop the flagged items that were never claims about the school.

    Sees the sentences and not the source. Whether something is a claim is a
    property of the sentence, and handing this pass the context invited it to
    second-guess support instead, which is what cost recall when the two jobs
    were one call.
    """
    if not claims:
        return []

    listing = "\n".join("%d. %s" % (i + 1, c) for i, c in enumerate(claims))
    response = client.chat.completions.create(
        model=GRADER_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{
            "role": "system",
            "content": CLAIM_FILTER_PROMPT
        }, {
            "role": "user",
            "content": "ITEMS:\n" + listing
        }])

    labels = json.loads(response.choices[0].message.content).get("labels", [])

    # A short or malformed reply must not silently discard findings, so
    # anything the filter did not label is kept.
    kept = []
    for i, claim in enumerate(claims):
        label = labels[i] if i < len(labels) else "school_fact"
        if str(label).strip() == "school_fact":
            kept.append(claim)
    return kept


# Tried and removed: a regex escape hatch that kept any claim carrying a
# titled personal name, an email, a clock time, a date or a sum of money,
# whatever the filter called it. The reasoning was that "emailed directly to
# the principal, Dr. Keith Morgan" is a referral in form and a fabrication in
# substance, and that one case was the difference between recall 1.000 and
# 0.889.
#
# It did not work. Precision fell 0.800 to 0.727 and recall did not move, so
# it readmitted false positives without recovering the case it was written
# for - meaning that case is lost somewhere other than the filter, and the
# escape hatch was solving a problem it had misidentified. Recorded rather
# than retried.


# Words too common to carry evidence either way.
STOPWORDS = frozenset("""a an and are as at be by can do does for from has have
how i if in is it its me my no not of on or our so that the their them there
these they this to was we were what when where which who will with you your
also please more some any all be been being at into over under""".split())

# How much of a claim's substance has to be findable in the source before the
# flag is treated as a mistake by the grader.
SUPPORT_THRESHOLD = 0.9


def lexically_supported(claim, context):
    """Whether every distinctive word of a claim already appears in the source.

    A backstop against the grader's precision falling apart on long contexts,
    which is a measured problem rather than a hypothetical one. Once ingest
    started reading linked documents the retrieved context roughly tripled in
    density, and gpt-4o-mini began reporting claims as unsupported whose every
    term was sitting in the text it had been given - "Completing the FAFSA is a
    graduation requirement", "Order your Cap, Gown & Tassel Unit from Herff
    Jones", both of them verbatim in context. The caution rate went 16% to 32%
    on a corpus that had got better, not worse.

    This does not try to decide whether the source *supports* the claim, which
    is genuinely hard and is the grader's job. It only catches the case where
    the source plainly contains the material: if every distinctive word is
    already there, the grader has almost certainly lost it in the noise rather
    than found a fabrication.

    Deliberately conservative. A near-total match is required, so a claim that
    keeps a topic's vocabulary while inverting its meaning - "students may NOT
    check out after 2:45" against a source saying they may - still reaches the
    reader as a flag, because that is a real failure this cannot detect.
    """
    words = {w for w in re.findall(r"[a-z0-9]+", claim.lower())
             if w not in STOPWORDS and len(w) > 2}
    if len(words) < 3:
        return False

    haystack = set(re.findall(r"[a-z0-9]+", context.lower()))
    found = sum(1 for w in words if w in haystack)
    return found / len(words) >= SUPPORT_THRESHOLD


def check_claim_grounding(client, answer, context):
    """Which claims about the school the context fails to support.

    Two passes. The first finds anything unsupported and the second decides
    which of those were claims at all, because one call asked to do both did
    each of them worse - see the note above DETECTOR_PROMPT.

    The filter only runs when the detector found something, so an answer with
    nothing wrong still costs a single call.
    """
    claims = detect_unsupported(client, answer, context)
    if not claims:
        return []

    claims = keep_real_claims(client, claims)

    # Drop anything the source plainly already contains. See
    # lexically_supported: on long document-derived contexts the grader
    # reports claims whose every word is sitting in the text it was handed.
    claims = [c for c in claims if not lexically_supported(c, context)]

    # Last, drop anything the answer does not actually say. The detector
    # sometimes reports a noun phrase describing what was missing - "senior
    # checklists", "exact drop-off locations" - rather than a sentence the
    # answer asserts. Those are not spans of the answer, so requiring the
    # match removes them without another model call.
    return [c for c in claims if quoted_from(c, answer)][:5]


def normalise_for_match(text):
    """Lowercase, collapse whitespace, drop punctuation that varies in quoting."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def quoted_from(claim, answer):
    """Whether the claim is really a span of the answer.

    Exact substring matching is too brittle: the grader reflows line breaks and
    normalises curly quotes, so a genuine quote can fail on punctuation alone.
    Matching on normalised text keeps those, and a short claim is additionally
    required to carry enough words to be a real assertion - a two-word
    fragment matching somewhere in the answer says nothing.
    """
    c = normalise_for_match(claim)
    if len(c.split()) < 4:
        return False
    return c in normalise_for_match(answer)


DECLINE_PROMPT = """Does this reply actually answer the question, or does it
decline because it does not have the information?

Answer "declined" when the substance of the reply is that it does not know, or
that the topic is not covered, and it states no facts that answer the question.
A reply that only points the reader at a page or the front office has declined.

Answer "answered" when it states facts that answer the question, even partly,
and even if it also suggests contacting the school.

Reply with JSON only: {"verdict": "answered"} or {"verdict": "declined"}"""


def answer_declined(client, question, answer):
    """Whether the reply gave up rather than answered.

    Not a keyword test. "I don't have the exact time" inside an otherwise
    complete answer is not a refusal, and "that isn't something the school
    pages cover" is one with none of the obvious markers, so a phrase list gets
    it wrong in both directions.
    """
    response = client.chat.completions.create(
        model=GRADER_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{
            "role": "system",
            "content": DECLINE_PROMPT
        }, {
            "role": "user",
            "content": "QUESTION:\n" + (question or "") + "\n\nREPLY:\n" + answer
        }])
    parsed = json.loads(response.choices[0].message.content)
    return parsed.get("verdict") == "declined"


def verify_answer(client, answer, context, stats, question=None):
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

    # A caution is for an answer that is confidently wrong. An answer that has
    # already told the reader it does not know is not confident, and stapling
    # "Only a loose match on the school pages" underneath it says the same
    # thing twice in a more doubtful voice - it reads as the bot being unsure
    # of its own honesty.
    #
    # So the retrieval-level warning is dropped when the reply declined, and
    # only then. If the model asserted something on a weak match, the warning
    # is exactly what it was built for and it stays.
    #
    # Checked last and only when it can change the outcome: retrieval must be
    # short of solid, and the two checks that find concrete problems must both
    # have come back clean. A reply with a fabricated link or an unsupported
    # claim keeps its notice whatever its tone.
    if (verdict.retrieval_level in ("weak", "none")
            and not verdict.bad_links
            and not verdict.unsupported):
        try:
            verdict.declined = answer_declined(client, question, answer)
        except Exception as e:
            # Staying quiet here would suppress a warning on the strength of a
            # call that did not happen. Failing back to showing it is the safe
            # direction.
            print("Decline check failed: " + str(e))
            verdict.declined = False

    return verdict
