"""
Precision and recall of the link-grounding check against a labeled fixture.

Offline and deterministic - no API key, no model call. The check is exact string
containment, so it should be perfect on cited URLs; the fixture exists to prove
that and to pin the edge cases (query strings, bare URLs, near-miss subpaths).

    python eval/link_grounding_eval.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hallucination  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures",
                       "link_cases.json")


def run():
    cases = json.load(open(FIXTURE, encoding="utf-8"))
    tp = fp = fn = 0
    failures = []

    for case in cases:
        got = set(hallucination.find_ungrounded_links(case["answer"],
                                                      case["context"]))
        want = set(case["expected_ungrounded"])
        tp += len(got & want)
        fp += len(got - want)
        fn += len(want - got)
        if got != want:
            failures.append((case["name"], sorted(want - got), sorted(got - want)))

    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    print(f"cases {len(cases)} | tp {tp} fp {fp} fn {fn}")
    print(f"precision {precision:.3f}  recall {recall:.3f}  f1 {f1:.3f}")
    if failures:
        print("\nfailures:")
        for name, missed, spurious in failures:
            print(f"  {name}")
            if missed:
                print(f"     missed:   {missed}")
            if spurious:
                print(f"     spurious: {spurious}")
    else:
        print("\nall cases pass")

    return precision, recall


if __name__ == "__main__":
    run()
