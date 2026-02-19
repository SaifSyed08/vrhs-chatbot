"""
Rebuild the index from the live site, and say whether anything actually moved.

Run by hand after a site change, and fortnightly by
.github/workflows/refresh-index.yml. The scheduled run is the point: a corpus
is a snapshot that quietly stops being true, without anything failing. A club
changes room, the announcements document rolls over, a bell schedule is
replaced - and the bot keeps answering confidently from the old copy. The
grounding checks cannot catch that, because the answer genuinely is supported
by the context. The context is just old.

Writes data/vrhs_embeddings.json, and touches .index-changed only when the
corpus differs in substance. Embeddings wobble in their last digits between
runs, so the file nearly always differs by a byte; comparing chunk text instead
is what tells a real change from float noise, and it is the difference between
a fortnightly commit that means something and one that means nothing.

    OPENAI_API_KEY=... python -c "import rebuild; rebuild.main()"
"""

import json
import os
import sys

os.environ.setdefault("VRHS_NO_WARM", "1")

import config  # noqa: E402
import main as app  # noqa: E402

MARKER = ".index-changed"


def corpus_signature(chunks):
    """What the corpus says, ignoring how the vectors came out.

    Sorted, because chunk order follows crawl order and a page answering a
    millisecond sooner is not a change worth committing.
    """
    return sorted((c.get("source", ""), c["text"]) for c in chunks)


def load_existing():
    try:
        with open(config.EMBEDDINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def summarise(chunks):
    kinds = {}
    for c in chunks:
        kind = c.get("kind", "prose")
        kinds[kind] = kinds.get(kind, 0) + 1
    return kinds


def main():
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")

    old = load_existing()
    old_signature = corpus_signature(old) if old else None

    print("crawling and reading linked files...")
    chunks = app.build_chunks()
    if not chunks:
        sys.exit("ingest produced no chunks - refusing to overwrite the index")

    # A collapse usually means the site was unreachable or its markup changed,
    # and overwriting a working index with the wreckage is worse than doing
    # nothing. Better to serve a fortnight-old corpus than an empty one.
    if old and len(chunks) < len(old) * 0.6:
        sys.exit("ingest produced %d chunks against %d before - too large a "
                 "drop to be a content change, refusing to overwrite"
                 % (len(chunks), len(old)))

    new_signature = corpus_signature(chunks)
    if old_signature == new_signature:
        print("corpus unchanged: %d chunks, %s" % (len(chunks),
                                                   summarise(chunks)))
        print("nothing to embed, nothing to commit")
        return

    print("corpus changed: %d chunks before, %d now"
          % (len(old) if old else 0, len(chunks)))
    print("by kind: %s" % summarise(chunks))

    vectors = app.embed_texts([c["text"] for c in chunks])
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector

    with open(config.EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    with open(MARKER, "w", encoding="utf-8") as f:
        f.write("changed\n")

    print("wrote %s" % config.EMBEDDINGS_PATH)
    print()
    print("The answer cache is now stale: it carries the old index's")
    print("fingerprint and main.py will refuse to serve it. Re-run prewarm.py")
    print("to rebuild it, or leave it and every question takes the live path.")


if __name__ == "__main__":
    main()
