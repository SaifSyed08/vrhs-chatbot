"""
Corpus hygiene for the scraped VRHS knowledge base.

Measured problem (see eval/corpus_stats.py): a naive full-page scrape of the
school site produced a corpus where 72.6% of all words sat inside navigation
boilerplate repeated across at least 60% of pages, and 15 of 91 chunks were
exact duplicates of another chunk.

The consequence is not wasted disk. It is that every chunk embeds mostly the
same nav menu, so the cosine similarity between two *unrelated* chunks has a
median of 0.88 - which destroys any absolute similarity threshold downstream.
This module removes the boilerplate before embedding.
"""

from collections import defaultdict

# A shingle repeated across at least this fraction of distinct source pages is
# navigation, not content. Chosen because the site's nav appears on every page
# while genuinely shared phrasing ("Vista Ridge High School") is short enough to
# fall below the shingle width.
BOILERPLATE_PAGE_RATIO = 0.6
SHINGLE_WIDTH = 5


def find_boilerplate_shingles(chunks, ratio=BOILERPLATE_PAGE_RATIO):
    """Return the set of word shingles that appear across most source pages."""
    pages = {c["source"] for c in chunks}
    if len(pages) < 2:
        return set()

    seen = defaultdict(set)
    for chunk in chunks:
        words = chunk["text"].split()
        for i in range(len(words) - SHINGLE_WIDTH + 1):
            shingle = " ".join(words[i:i + SHINGLE_WIDTH])
            seen[shingle].add(chunk["source"])

    cutoff = ratio * len(pages)
    return {s for s, srcs in seen.items() if len(srcs) >= cutoff}


def strip_boilerplate(chunks):
    """Drop words covered by cross-page shingles, keeping per-page content."""
    boiler = find_boilerplate_shingles(chunks)
    if not boiler:
        return chunks

    cleaned = []
    for chunk in chunks:
        # Link chunks are deliberately short and already deduplicated by URL,
        # so the length rule below would throw them all away.
        if chunk.get("kind") == "link":
            cleaned.append(chunk)
            continue

        words = chunk["text"].split()
        covered = [False] * len(words)
        for i in range(len(words) - SHINGLE_WIDTH + 1):
            if " ".join(words[i:i + SHINGLE_WIDTH]) in boiler:
                for j in range(i, i + SHINGLE_WIDTH):
                    covered[j] = True

        kept = " ".join(w for w, hit in zip(words, covered) if not hit)

        # A chunk that was entirely nav has nothing left worth embedding.
        if len(kept.split()) >= 10:
            cleaned.append({**chunk, "text": kept})

    return cleaned


def dedupe(chunks):
    """Remove chunks whose text is byte-identical to one already kept."""
    seen, out = set(), []
    for chunk in chunks:
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            out.append(chunk)
    return out


def prepare(chunks):
    """Full hygiene pass, run between scraping and embedding."""
    before = len(chunks)
    chunks = dedupe(strip_boilerplate(chunks))
    print(f"corpus: {before} chunks -> {len(chunks)} after boilerplate strip "
          f"and dedupe")
    return chunks
