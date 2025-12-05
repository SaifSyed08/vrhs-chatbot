"""
Re-embed the knowledge base with the corpus hygiene pass applied, so the
before/after retrieval comparison in README.md is reproducible without
re-scraping the school site.

Takes the chunk texts already in data/vrhs_embeddings.json, runs them through
corpus.prepare (the same call main.py now makes between scraping and
embedding), re-embeds the survivors, and writes a second corpus file. Both are
committed so eval/retrieval_eval.py can be pointed at either.

    OPENAI_API_KEY=... python eval/rebuild_clean_corpus.py
"""

import json
import os
import sys

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import corpus  # noqa: E402

SOURCE = "data/vrhs_embeddings.json"
TARGET = "data/vrhs_embeddings_clean.json"
EMBED_MODEL = "text-embedding-ada-002"


def main():
    docs = json.load(open(SOURCE, encoding="utf-8"))
    chunks = [{"text": d["text"], "source": d["source"]} for d in docs]

    cleaned = corpus.prepare(chunks)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for i, chunk in enumerate(cleaned, 1):
        response = client.embeddings.create(model=EMBED_MODEL,
                                            input=chunk["text"])
        chunk["embedding"] = response.data[0].embedding
        print(f"  embedded {i}/{len(cleaned)}", end="\r")

    with open(TARGET, "w", encoding="utf-8") as f:
        json.dump(cleaned, f)
    print(f"\nwrote {TARGET} ({len(cleaned)} chunks)")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    main()
