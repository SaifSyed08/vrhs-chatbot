"""
Head-to-head comparison of the three claim-grounding architectures in
detectors.py, over the labeled set in fixtures/grounding_cases.json.

Reports precision, recall, F1, median latency and cost per check. The two
threshold-based detectors are swept and reported at their best F1, which is the
most favourable reading of them - if the LLM judge still wins on a set tuned
against it, the margin is real.

    OPENAI_API_KEY=... python eval/detection_ablation.py
"""

import json
import os
import statistics
import sys
import time

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detectors  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "grounding_cases.json")

# Published rates, USD per 1M tokens, for the cost column.
ADA_PER_M = 0.10
MINI_IN_PER_M, MINI_OUT_PER_M = 0.15, 0.60


def score(flags, labels):
    tp = sum(f and l for f, l in zip(flags, labels))
    fp = sum(f and not l for f, l in zip(flags, labels))
    fn = sum((not f) and l for f, l in zip(flags, labels))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def sweep(scores, labels, lower_is_hallucinated=True):
    """Best F1 over all thresholds, plus the threshold that achieved it."""
    best = (0.0, None, 0.0, 0.0)
    for t in sorted(set(scores)):
        flags = [s < t if lower_is_hallucinated else s > t for s in scores]
        p, r, f1 = score(flags, labels)
        if f1 > best[0]:
            best = (f1, t, p, r)
    return best


def run():
    cases = json.load(open(FIXTURE, encoding="utf-8"))
    labels = [c["hallucinated"] for c in cases]
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"cases {len(cases)} | hallucinated {sum(labels)} | "
          f"grounded {len(labels) - sum(labels)}\n")
    rows = []

    # 1. lexical overlap - no API call
    scores, times = [], []
    for c in cases:
        t0 = time.perf_counter()
        _, detail = detectors.lexical_overlap(c["answer"], c["context"])
        times.append((time.perf_counter() - t0) * 1000)
        scores.append(detail["score"])
    f1, thr, p, r = sweep(scores, labels)
    rows.append(("lexical overlap", p, r, f1, statistics.median(times), 0.0,
                 f"threshold {thr:.2f}"))

    # 2. embedding similarity - one embedding call
    scores, times, tokens = [], [], 0
    for c in cases:
        t0 = time.perf_counter()
        _, detail = detectors.embedding_similarity(client, c["answer"],
                                                   c["context"])
        times.append((time.perf_counter() - t0) * 1000)
        scores.append(detail["score"])
        tokens += len(c["context"].split()) + len(c["answer"].split())
    f1, thr, p, r = sweep(scores, labels)
    embed_cost = (tokens * 1.3 / 1e6) * ADA_PER_M / len(cases)
    rows.append(("embedding similarity", p, r, f1, statistics.median(times),
                 embed_cost, f"threshold {thr:.3f}"))

    # 3. LLM judge - one gpt-4o-mini call
    flags, times, in_tok, out_tok = [], [], 0, 0
    for c in cases:
        t0 = time.perf_counter()
        flagged, detail = detectors.llm_judge(client, c["answer"], c["context"])
        times.append((time.perf_counter() - t0) * 1000)
        flags.append(flagged)
        in_tok += (len(c["context"].split()) + len(c["answer"].split())) * 1.3 + 180
        out_tok += 40
    p, r, f1 = score(flags, labels)
    judge_cost = ((in_tok / 1e6) * MINI_IN_PER_M +
                  (out_tok / 1e6) * MINI_OUT_PER_M) / len(cases)
    rows.append(("llm judge (gpt-4o-mini)", p, r, f1, statistics.median(times),
                 judge_cost, "no threshold"))

    # 4. cascade - lexical prefilter, LLM judge only for the ambiguous band
    flags, times, escalated = [], [], 0
    for c in cases:
        t0 = time.perf_counter()
        flagged, detail = detectors.cascade(client, c["answer"], c["context"])
        times.append((time.perf_counter() - t0) * 1000)
        flags.append(flagged)
        escalated += detail["escalated"]
    p, r, f1 = score(flags, labels)
    rows.append(("cascade (lexical -> llm)", p, r, f1, statistics.median(times),
                 judge_cost * escalated / len(cases),
                 f"{escalated}/{len(cases)} escalated"))

    # 5. chain-of-verification - plan verification questions, then answer them
    flags, times, cove_in, cove_out = [], [], 0, 0
    for c in cases:
        t0 = time.perf_counter()
        flagged, _ = detectors.chain_of_verification(client, c["answer"],
                                                     c["context"])
        times.append((time.perf_counter() - t0) * 1000)
        flags.append(flagged)
        words = len(c["context"].split()) + len(c["answer"].split())
        cove_in += words * 1.3 + 340   # two calls, two system prompts
        cove_out += 130
    p, r, f1 = score(flags, labels)
    cove_cost = ((cove_in / 1e6) * MINI_IN_PER_M +
                 (cove_out / 1e6) * MINI_OUT_PER_M) / len(cases)
    rows.append(("chain-of-verification", p, r, f1, statistics.median(times),
                 cove_cost, "2 calls per check"))

    header = f"{'architecture':26}{'prec':>7}{'rec':>7}{'F1':>7}{'p50 ms':>9}{'$/check':>11}  notes"
    print(header)
    print("-" * len(header))
    for name, p, r, f1, ms, cost, note in rows:
        print(f"{name:26}{p:7.3f}{r:7.3f}{f1:7.3f}{ms:9.0f}{cost:11.6f}  {note}")

    print("\nthreshold detectors are shown at their best achievable F1, which "
          "overstates them:\nthat threshold was chosen on this same 20-case set.")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    run()
