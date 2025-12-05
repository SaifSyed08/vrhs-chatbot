"""
End-to-end retrieval and gating evaluation.

Requires OPENAI_API_KEY - it embeds each question for real. Measures the two
things the offline evals cannot:

  1. Retrieval accuracy: does the correct source page land in the top 3?
  2. Gate calibration: how far the standard-score margin separates questions the
     corpus can answer from questions it cannot. Z_SOLID and Z_WEAK in
     hallucination.py started as a pseudo-query lower bound; this replaces that
     guess with a measurement over real questions.

Deliberately does not import main.py - that would drag in flask and bs4 for a
script that only needs an embedding call.

    OPENAI_API_KEY=... python eval/retrieval_eval.py
"""

import json
import os
import sys

import numpy as np
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hallucination  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "questions.json")
EMBEDDINGS = os.environ.get(
    "VRHS_EMBEDDINGS",
    os.path.join(os.path.dirname(HERE), "data", "vrhs_embeddings.json"))
EMBED_MODEL = "text-embedding-ada-002"


def probe(client, matrix, sources, question):
    """Retrieve for one question, returning (stats, sources of the top 3)."""
    response = client.embeddings.create(model=EMBED_MODEL, input=question)
    q = np.array(response.data[0].embedding, dtype=np.float64)
    q /= np.linalg.norm(q)

    sims = matrix @ q
    order = np.argsort(sims)[::-1]
    spread = float(sims.std())
    stats = {
        "top": float(sims.max()),
        "z": float((sims.max() - sims.mean()) / spread) if spread else None,
    }
    return stats, [sources[i] for i in order[:3]]


def run():
    fx = json.load(open(FIXTURE, encoding="utf-8"))
    docs = json.load(open(EMBEDDINGS, encoding="utf-8"))

    matrix = np.array([d["embedding"] for d in docs], dtype=np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    sources = [d["source"] for d in docs]

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"corpus: {os.path.basename(EMBEDDINGS)} ({len(docs)} chunks)\n")

    hit1 = hit3 = 0
    z_answerable, z_oos = [], []
    top_answerable, top_oos = [], []

    print("-- answerable --")
    for case in fx["answerable"]:
        stats, srcs = probe(client, matrix, sources, case["question"])
        want = case["expect_source_contains"]
        hit1 += want in srcs[0]
        top3 = any(want in s for s in srcs)
        hit3 += top3
        z_answerable.append(stats["z"])
        top_answerable.append(stats["top"])
        print(f"  {'ok ' if top3 else 'MISS'} z={stats['z']:5.2f} "
              f"top={stats['top']:.3f}  {case['question'][:50]}")

    print("\n-- out of scope (the corpus cannot answer these) --")
    for case in fx["out_of_scope"]:
        stats, _ = probe(client, matrix, sources, case["question"])
        z_oos.append(stats["z"])
        top_oos.append(stats["top"])
        print(f"      z={stats['z']:5.2f} top={stats['top']:.3f}  "
              f"{case['question'][:50]}")

    n = len(fx["answerable"])
    a, o = np.array(z_answerable), np.array(z_oos)
    ta, to = np.array(top_answerable), np.array(top_oos)
    print(f"\nretrieval: top-1 {hit1}/{n} ({hit1 / n:.1%})   "
          f"top-3 {hit3}/{n} ({hit3 / n:.1%})")
    print(f"z answerable    mean {a.mean():5.2f}  p5  {np.percentile(a, 5):5.2f}")
    print(f"z out-of-scope  mean {o.mean():5.2f}  p95 {np.percentile(o, 95):5.2f}")

    print(f"raw top answerable   min {ta.min():.3f}  mean {ta.mean():.3f}")
    print(f"raw top out-of-scope max {to.max():.3f}  mean {to.mean():.3f}")

    # Which signal actually separates answerable from unanswerable questions?
    print("\nseparating power:")
    for label, pos, neg in (("raw top-1 cosine", ta, to), ("z margin", a, o)):
        grid = np.linspace(min(pos.min(), neg.min()), max(pos.max(), neg.max()),
                           800)
        acc = [((pos >= t).sum() + (neg < t).sum()) / (len(pos) + len(neg))
               for t in grid]
        best = float(grid[int(np.argmax(acc))])
        print(f"  {label:18} best cutoff {best:.3f} -> accuracy {max(acc):.1%}")

    print(f"\nconfigured: SIMILARITY_SOLID={hallucination.SIMILARITY_SOLID} "
          f"SIMILARITY_WEAK={hallucination.SIMILARITY_WEAK}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set - this eval embeds questions for real")
    run()
