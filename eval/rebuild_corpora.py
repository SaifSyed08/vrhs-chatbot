"""
Rebuild both corpora from a single scrape.

Writes the production knowledge base (hygiene pass applied, what main.py reads)
and a baseline corpus with the pass skipped, so the before/after comparison in
README.md is reproducible without scraping the school site twice.

    OPENAI_API_KEY=... python eval/rebuild_corpora.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import corpus  # noqa: E402

PRODUCTION = "data/vrhs_embeddings.json"
BASELINE = "data/vrhs_embeddings_baseline.json"


def embed_all(main, chunks, label):
    for i, chunk in enumerate(chunks, 1):
        chunk["embedding"] = main.embed_text(chunk["text"])
        print(f"  {label}: embedded {i}/{len(chunks)}", end="\r")
    print()
    return chunks


def run():
    import main

    scraped = main.scrape_vrhs_pages()
    manual = [{"text": t, "source": "manual"} for t in main.MANUAL_CHUNKS]

    baseline = [dict(c) for c in scraped] + [dict(c) for c in manual]
    production = corpus.prepare([dict(c) for c in scraped]) + \
        [dict(c) for c in manual]

    print(f"baseline {len(baseline)} chunks | production {len(production)} chunks")

    for path, chunks, label in ((BASELINE, baseline, "baseline"),
                                (PRODUCTION, production, "production")):
        embed_all(main, chunks, label)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f)
        print(f"  wrote {path}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    run()
