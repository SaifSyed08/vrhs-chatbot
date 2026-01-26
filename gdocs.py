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

import logging
import re
import urllib.parse

log = logging.getLogger("vrhs.gdocs")

DOC_ID = re.compile(r"https://docs\.google\.com/document/d/([\w-]+)")
DRIVE_ID = re.compile(r"https://drive\.google\.com/file/d/([\w-]+)")
SHEET_ID = re.compile(r"https://docs\.google\.com/spreadsheets/d/([\w-]+)")

TIMEOUT = 25

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


def document_text(url, session=None):
    """Plain text of a public Google Doc or Sheet, or None."""
    import requests

    fid = document_id(url)
    export = None
    if fid:
        export = ("https://docs.google.com/document/d/%s/export?format=txt"
                  % fid)
    else:
        fid = sheet_id(url)
        if fid:
            export = ("https://docs.google.com/spreadsheets/d/%s/export"
                      "?format=csv" % fid)
    if not export:
        return None

    get = (session or requests).get
    try:
        response = get(export, timeout=TIMEOUT)
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


def linked_documents(links):
    """Chunk-ready text for every public Google Doc the site links to.

    Returns (label, url, text) triples. Order follows the links so a rerun is
    reproducible, and each unique URL is fetched once however often it is
    linked.
    """
    import requests

    session = requests.Session()
    done, out = set(), []

    for label, href, _source in links:
        if len(out) >= MAX_DOCS:
            log.info("stopping at %d documents", MAX_DOCS)
            break
        if href in done or not (document_id(href) or sheet_id(href)):
            continue
        done.add(href)

        log.info("reading document: %s", (label or href)[:60])
        text = document_text(href, session)
        if text:
            out.append((label or "Linked document", href, text))
        else:
            log.info("  skipped, not public or no text")

    session.close()
    log.info("read %d of %d linked documents", len(out), len(done))
    return out
