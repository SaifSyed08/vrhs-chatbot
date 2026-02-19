"""
Read the Google files the site links to, instead of only linking them.

A lot of what students actually want is not on the school website. It is in a
Google Doc or a Drive PDF that the site links to and the scraper walked past,
because the crawl is same-domain and skips binary files. Bell times, the A/B
calendar, the senior checklist - all one hop away and all invisible.

Two things are fetched here, and they solve different problems.

**Document text.** A public Google Doc exports as plain text from a URL, no API
key and no credentials. 16 of the 23 docs linked from the site come back with
real content that way. Those become chunks like any page, so the bot can answer
from what the document says rather than handing over the link and wishing the
reader luck.

**Drive filenames.** A Drive PDF cannot be read this cheaply, and it turns out
it does not need to be. The site links two files both labelled "A/B Calendar",
one of which is last year's; the model picks between them blind and picked
wrong. The filename settles it outright:

    1xX13...  ->  "2025-2026 District A_B Calendar.pdf"
    1n54N...  ->  "VRHS 2026-2027 Calendar.pdf"

That arrives in a Content-Disposition header on a request that does not even
have to finish. Attaching it to the link label lets the existing preference for
year-bearing labels do the rest, and it fixed the wrong-year link without
parsing a single page of PDF. The one that would have needed OCR - an
image-only export with no text layer - is exactly the one whose filename says
what it is.

Everything here fails soft. A document that will not load must not take an
ingest down with it, and a missing filename just leaves a label as it was.
"""

import csv
import io
import logging
import re
import urllib.parse

log = logging.getLogger("vrhs.gdocs")

DOC_ID = re.compile(r"https://docs\.google\.com/document/d/([\w-]+)")
DRIVE_ID = re.compile(r"https://drive\.google\.com/file/d/([\w-]+)")
# Matches a sheet however it is linked or embedded. The clubs page carries its
# spreadsheet as an iframe ending /htmlembed, and published sheets use a /d/e/
# path, so keying on "/spreadsheets/d/" alone missed both.
SHEET_ID = re.compile(
    r"https://docs\.google\.com/spreadsheets/d/(?:e/)?([\w-]+)")

TIMEOUT = 25
# Sheets take longer than documents. The clubs spreadsheet is 45 KB of CSV and
# timed out at 25 s, which is how it stayed missing from the corpus while
# appearing to be "not public".
SHEET_TIMEOUT = 60

# Bounds. Ingest already takes about twenty seconds and these are sequential
# network calls on top of it, so the crawl's habit of staying small applies
# here too.
MAX_DOCS = 40
MIN_USEFUL_CHARS = 120

# A document that exports as a login page or a consent interstitial is not
# public, whatever the status code said.
NOT_PUBLIC = ("<html", "<!doctype", "accounts.google.com")


def document_id(url):
    m = DOC_ID.search(url or "")
    return m.group(1) if m else None


def drive_id(url):
    m = DRIVE_ID.search(url or "")
    return m.group(1) if m else None


def sheet_id(url):
    m = SHEET_ID.search(url or "")
    return m.group(1) if m else None


def remote_name(url, session=None):
    """The real filename behind a Drive link, or None.

    Streamed and closed immediately: the header is the whole point and some of
    these files are several megabytes.
    """
    import requests

    fid = drive_id(url)
    if not fid:
        return None

    get = (session or requests).get
    try:
        response = get("https://drive.google.com/uc?export=download&id=" + fid,
                       timeout=TIMEOUT, stream=True)
        disposition = response.headers.get("content-disposition", "")
        response.close()
    except Exception as e:
        log.warning("could not read filename for %s: %s", fid[:12], e)
        return None

    match = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", disposition)
    if not match:
        return None

    name = urllib.parse.unquote(match.group(1)).strip()
    # The extension is noise in a link label.
    return re.sub(r"\.(pdf|docx?|xlsx?|pptx?)$", "", name, flags=re.I)


def sheet_rows(text, label):
    """One readable line per spreadsheet row, headed by its first column.

    A sheet is not prose and chunking it by word count cuts rows in half, which
    leaves a club's name in one chunk and its meeting time in another. Row by
    row, each club becomes a unit that can be retrieved on its own and read
    whole - which is what "when does Aerospace Club meet" actually needs.
    """
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return []

    header = [h.strip() for h in rows[0]]
    out = []

    for row in rows[1:]:
        cells = [c.strip() for c in row]
        if not any(cells):
            continue

        name = cells[0] if cells else ""
        # A row with only its first cell filled is a section banner, not an
        # entry - "STUDENT INTEREST CLUBS" and the like.
        if name and not any(cells[1:]):
            continue

        parts = []
        for key, value in zip(header[1:], cells[1:]):
            if value and key:
                parts.append("%s: %s" % (key, value))
        if not parts:
            continue

        out.append("%s, from %s. %s" % (name or "Entry", label,
                                        " ".join(parts)))
    return out


def document_text(url, session=None):
    """Plain text of a public Google Doc or Sheet, or None."""
    import requests

    fid = document_id(url)
    export, timeout = None, TIMEOUT
    if fid:
        export = ("https://docs.google.com/document/d/%s/export?format=txt"
                  % fid)
    else:
        fid = sheet_id(url)
        if fid:
            export = ("https://docs.google.com/spreadsheets/d/%s/export"
                      "?format=csv" % fid)
            timeout = SHEET_TIMEOUT
    if not export:
        return None

    get = (session or requests).get
    try:
        response = get(export, timeout=timeout)
    except Exception as e:
        log.warning("could not read doc %s: %s", fid[:12], e)
        return None

    if response.status_code != 200:
        return None

    text = response.text or ""
    head = text[:300].lower()
    if any(marker in head for marker in NOT_PUBLIC):
        return None

    text = text.replace("﻿", "").strip()
    if len(text) < MIN_USEFUL_CHARS:
        # Docs that export to almost nothing are usually a page of images, or
        # a shell whose content lives in a linked sub-document.
        return None

    # Newlines are load-bearing in a CSV and noise in a document. Collapsing
    # them for everything turned the clubs spreadsheet into one 45,000
    # character line, which csv.reader read as a single row with no data in it.
    if sheet_id(url) and not document_id(url):
        return text
    return " ".join(text.split())


def enrich_labels(links):
    """Rewrite Drive link labels to carry the real filename.

    Takes and returns the scraper's (label, href, source) triples. A label that
    already names a year is left alone - the site said something specific and
    guessing over it would be worse.
    """
    import requests

    session = requests.Session()
    seen, out = {}, []

    for label, href, source in links:
        if not drive_id(href) or re.search(r"(19|20)\d\d", label or ""):
            out.append((label, href, source))
            continue

        if href not in seen:
            log.info("reading filename: %s", (label or href)[:60])
            seen[href] = remote_name(href, session)
        name = seen[href]

        if name and name.lower() != (label or "").lower():
            # Both kept. The site's label is what a reader recognises, the
            # filename is what disambiguates two of them.
            out.append(("%s (%s)" % (label, name), href, source))
        else:
            out.append((label, href, source))

    session.close()
    return out


WORDS_PER_CHUNK = 150

# Anchor text that names nothing. A file linked as "click here" and embedded
# under a real heading should be called by the heading.
VAGUE_LABELS = frozenset([
    "click here", "here", "link", "this link", "view", "open", "download",
    "more", "read more", "see here", "click", "this", "document", "doc",
    "click here to open to print or download", "print", "form",
])


def file_key(url):
    """Identifies the file behind a URL, whatever shape the URL is.

    The clubs spreadsheet is linked as an anchor reading "click here" and
    embedded as an iframe under the heading "Clubs". Same file, two URLs -
    /edit and /htmlembed - so keying on the URL read it twice, chunked it
    twice, and offered a reader two source pills for one document, one of them
    labelled "click here".
    """
    return document_id(url) or sheet_id(url) or drive_id(url) or url


def label_quality(label):
    """How much a label tells you. Higher is better."""
    clean = " ".join((label or "").split())
    if not clean or clean.lower() in VAGUE_LABELS:
        return (0, 0)
    return (1, len(clean))


def readable(url):
    """Whether this is a Google file this module knows how to open."""
    return bool(document_id(url) or sheet_id(url))


def linked_documents(references):
    """Chunk-ready pieces for every public Google file the site points at.

    Takes (label, url) pairs from anywhere the crawl found a Google URL -
    anchors, iframes, embeds - and returns (label, url, pieces). Nothing here
    knows about any particular document: it matches on the URL shape, so a
    spreadsheet embedded on a page nobody has thought about yet is read on the
    next ingest without anyone adding it to a list.

    Pieces are split by format rather than uniformly. A document gets word
    windows like a page; a spreadsheet gets one piece per row, because chunking
    a sheet by word count cuts rows in half and leaves a club's name in one
    chunk and its meeting time in another.
    """
    import requests

    import requests as _requests  # noqa: F401  (kept for symmetry)

    # One entry per file, keeping the most informative label anyone gave it and
    # the URL that came with it.
    best = {}
    for label, href in references:
        if not href or not readable(href):
            continue
        key = file_key(href)
        current = best.get(key)
        if current is None or label_quality(label) > label_quality(current[0]):
            best[key] = (label, href)

    session = requests.Session()
    done, out = set(), []

    for label, href in best.values():
        if len(out) >= MAX_DOCS:
            log.info("stopping at %d documents", MAX_DOCS)
            break
        done.add(href)

        name = " ".join((label or "").split()) or "Linked document"
        if name.lower() in VAGUE_LABELS:
            name = "Linked document"
        log.info("reading %s: %s",
                 "spreadsheet" if sheet_id(href) and not document_id(href)
                 else "document", name[:60])

        text = document_text(href, session)
        if not text:
            log.info("  skipped, not public or no text")
            continue

        if sheet_id(href) and not document_id(href):
            pieces = sheet_rows(text, name)
        else:
            words = text.split()
            pieces = [" ".join(words[i:i + WORDS_PER_CHUNK])
                      for i in range(0, len(words), WORDS_PER_CHUNK)]

        pieces = [p for p in pieces if p.strip()]
        if pieces:
            out.append((name, href, pieces))
            log.info("  %d pieces", len(pieces))

    session.close()
    log.info("read %d of %d linked files", len(out), len(done))
    return out
