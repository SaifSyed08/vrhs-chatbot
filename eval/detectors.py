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
