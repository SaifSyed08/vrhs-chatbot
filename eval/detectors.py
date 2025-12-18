"""
Candidate architectures for the claim-grounding check, so the production choice
is a measured one rather than the first thing that worked.

All three answer the same question - does this answer assert something the
retrieved context does not support - at very different cost points:

  lexical_overlap      no API call, pure string work
  embedding_similarity one embedding call for the answer; context vectors are
                       already in the index, so they are free at query time
  llm_judge            one gpt-4o-mini call, the current production check

Each returns (flagged, detail).
"""

import json
import re

import numpy as np

SENTENCE = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[a-z0-9]+")

# Words too common to carry grounding signal.
STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "to", "of", "in", "on", "at", "for", "with", "by", "from", "as",
    "that", "this", "these", "those", "it", "its", "you", "your", "they",
    "their", "will", "can", "may", "must", "should", "would", "there", "here",
    "have", "has", "had", "if", "not", "no", "any", "all", "each", "per",
}


def sentences(text):
    return [s.strip() for s in SENTENCE.split(text.strip()) if len(s.strip()) > 12]


def content_words(text):
    return {w for w in WORD.findall(text.lower()) if len(w) > 3 and w not in STOP}


def lexical_overlap(answer, context, threshold=0.5):
    """Flag if any sentence shares too few content words with the context."""
    ctx = content_words(context)
    worst, worst_sentence = 1.0, ""

    for sentence in sentences(answer) or [answer]:
        words = content_words(sentence)
        if not words:
            continue
        covered = len(words & ctx) / len(words)
        if covered < worst:
            worst, worst_sentence = covered, sentence

    return worst < threshold, {"score": worst, "sentence": worst_sentence}


def embedding_similarity(client, answer, context, threshold=0.85,
                         model="text-embedding-ada-002"):
    """Flag if any answer sentence is far from every context sentence.

    Context sentences are embedded here for the experiment. In production the
    chunk vectors already exist in the index, so only the answer costs anything.
    """
    ctx_sentences = sentences(context) or [context]
    ans_sentences = sentences(answer) or [answer]

    response = client.embeddings.create(model=model,
                                        input=ctx_sentences + ans_sentences)
    vectors = np.array([d.embedding for d in response.data], dtype=np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    ctx_vecs = vectors[:len(ctx_sentences)]
    ans_vecs = vectors[len(ctx_sentences):]

    best_per_sentence = (ans_vecs @ ctx_vecs.T).max(axis=1)
    worst = float(best_per_sentence.min())
    return worst < threshold, {"score": worst}


def llm_judge(client, answer, context, model="gpt-4o-mini"):
    """The production check: a second model reports unsupported claims."""
    import hallucination

    claims = hallucination.check_claim_grounding(client, answer, context)
    return bool(claims), {"claims": claims}


def cascade(client, answer, context, low=0.55, high=0.85):
    """Lexical prefilter, escalating only the ambiguous middle to the LLM judge.

    An answer that reuses almost none of the context's vocabulary is nearly
    always ungrounded, and one that reuses nearly all of it is nearly always
    grounded. Only the band between them needs a model call, which is where the
    cost sits.

    Returns (flagged, detail); detail["escalated"] records whether the LLM ran.
    """
    _, detail = lexical_overlap(answer, context, threshold=low)
    score = detail["score"]

    if score < low:
        return True, {"score": score, "escalated": False}
    if score >= high:
        return False, {"score": score, "escalated": False}

    flagged, judged = llm_judge(client, answer, context)
    return flagged, {"score": score, "escalated": True, **judged}


COVE_PLAN = """You are checking a chatbot answer for unsupported claims.

Break the ANSWER into its individual factual claims and write one short
verification question for each. Ask only about facts: names, numbers, dates,
times, requirements, locations, procedures. Ignore greetings, offers to help and
closing pleasantries.

Reply with JSON only: {"questions": ["...", "..."]}
At most 4 questions. Use an empty list if the answer states no facts."""

COVE_EXECUTE = """Answer each question using ONLY the SOURCE text.

For each question reply "supported" if SOURCE states the answer, or
"unsupported" if SOURCE does not state it. Judge against SOURCE alone, never
against your own knowledge, and treat a partial or approximate match as
unsupported.

Reply with JSON only:
{"verdicts": [{"question": "...", "status": "supported|unsupported"}]}"""


def chain_of_verification(client, answer, context, model="gpt-4o-mini"):
    """Chain-of-Verification (Dhuliawala et al., 2023), used as a detector.

    The published method plans verification questions, answers them
    independently of the original response, and rewrites the answer from the
    results. Here only the detection half is needed, so it runs the plan and
    execute stages and flags the answer if any verification comes back
    unsupported.

    The premise is that decomposing into narrow questions is harder to wave
    through than judging a whole paragraph at once. It costs two model calls
    instead of one.

    On this corpus it did not pay off. Decomposition produces questions about
    incidental details, and demanding the source state each one turns
    paraphrase into a false positive: precision 0.615 against the judge's
    1.000. A second run with the execute prompt loosened to accept paraphrase
    moved precision to 0.636 but cost recall, for a slightly worse F1 (0.700 vs
    0.727), so the stricter wording is kept. See eval/results.
    """
    plan = client.chat.completions.create(
        model=model, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": COVE_PLAN},
                  {"role": "user", "content": "ANSWER:\n" + answer}])
    questions = json.loads(plan.choices[0].message.content).get("questions", [])
    questions = [str(q).strip() for q in questions if str(q).strip()][:4]

    if not questions:
        return False, {"questions": [], "unsupported": []}

    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    execute = client.chat.completions.create(
        model=model, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": COVE_EXECUTE},
                  {"role": "user",
                   "content": "SOURCE:\n" + context + "\n\nQUESTIONS:\n" + numbered}])
    verdicts = json.loads(execute.choices[0].message.content).get("verdicts", [])

    unsupported = [v.get("question", "") for v in verdicts
                   if str(v.get("status", "")).lower().startswith("unsup")]
    return bool(unsupported), {"questions": questions, "unsupported": unsupported}
