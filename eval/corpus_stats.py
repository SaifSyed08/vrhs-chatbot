"""
Corpus and threshold measurements. Offline - reads only the committed
embeddings, needs no API key, and reproduces every corpus number in README.md.

    python eval/corpus_stats.py
"""

import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import corpus  # noqa: E402

EMBEDDINGS = "data/vrhs_embeddings.json"


def unit(docs):
    E = np.array([d["embedding"] for d in docs], dtype=np.float64)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def background(docs):
    """Cosine similarity of every chunk pair - the corpus noise floor."""
    S = unit(docs) @ unit(docs).T
    return S[np.triu_indices(len(docs), k=1)]


def report(docs):
    print(f"chunks {len(docs)} | dim {len(docs[0]['embedding'])} | "
          f"sources {len({d['source'] for d in docs})}")

    bg = background(docs)
    print("\n-- background similarity (unrelated chunk pairs) --")
    for p in (5, 25, 50, 75, 95):
        print(f"   p{p:<3} {np.percentile(bg, p):.4f}")
    print(f"   mean {bg.mean():.4f}   n_pairs {len(bg)}")

    print("\n-- duplicates --")
    counts = Counter(d["text"] for d in docs)
    dupes = {t: n for t, n in counts.items() if n > 1}
    print(f"   duplicate groups {len(dupes)} | "
          f"redundant chunks {sum(dupes.values()) - len(dupes)}")

    print("\n-- boilerplate --")
    site = [d for d in docs if d["source"] != "manual"]
    boiler = corpus.find_boilerplate_shingles(site)
    total = covered = 0
    for d in site:
        words = d["text"].split()
        total += len(words)
        hit = [False] * len(words)
        for i in range(len(words) - corpus.SHINGLE_WIDTH + 1):
            if " ".join(words[i:i + corpus.SHINGLE_WIDTH]) in boiler:
                for j in range(i, i + corpus.SHINGLE_WIDTH):
                    hit[j] = True
        covered += sum(hit)
    print(f"   cross-page shingles {len(boiler)} | "
          f"words in boilerplate {covered}/{total} ({covered / total:.1%})")

    print("\n-- can an absolute threshold work? --")
    S = unit(docs) @ unit(docs).T
    np.fill_diagonal(S, -np.inf)
    best = np.array([np.delete(S[i], i).max() for i in range(len(docs))])
    print(f"   perfect-match top-1, p5 : {np.percentile(best, 5):.4f}")
    print(f"   unrelated background, p50: {np.median(bg):.4f}")
    verdict = ("NO - distributions overlap"
               if np.percentile(best, 5) < np.median(bg) else "yes")
    print(f"   separable by a single cutoff? {verdict}")

    print("\n-- standard-score margin instead --")
    z = np.array([(np.delete(S[i], i).max() - np.delete(S[i], i).mean()) /
                  np.delete(S[i], i).std() for i in range(len(docs))])
    for p in (5, 25, 50, 95):
        print(f"   p{p:<3} z {np.percentile(z, p):.2f}")
    print(f"   min z {z.min():.2f}  (pseudo-query lower bound -> Z_SOLID)")

    print("\n-- effect of the hygiene pass --")
    plain = [{"text": d["text"], "source": d["source"]} for d in docs]
    after = corpus.prepare(plain)
    w_before = sum(len(d["text"].split()) for d in plain)
    w_after = sum(len(d["text"].split()) for d in after)
    print(f"   chunks {len(docs)} -> {len(after)} | "
          f"words {w_before} -> {w_after} ({1 - w_after / w_before:.1%} removed)")
    print("   note: background similarity after stripping requires re-embedding "
          "(run /embed), since the stored vectors encode the boilerplate.")


if __name__ == "__main__":
    if not os.path.exists(EMBEDDINGS):
        sys.exit(f"missing {EMBEDDINGS} - run the /embed route first")
    report(json.load(open(EMBEDDINGS, encoding="utf-8")))
