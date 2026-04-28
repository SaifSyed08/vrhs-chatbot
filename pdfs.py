"""
Read the PDFs the site links, instead of only naming them.

The school publishes a lot as PDF on Drive: the bell schedules, the detention
rules, the hours-owed form, the resource one-pagers, and both songs. Ingest
already learned the *filenames* of those - that is what fixed the two rival
"A/B Calendar" links - but it never opened one, so the bot could hand over a
link and nothing else.

Asked for the fight song it gave the link. Asked for the lyrics it said it did
not have them. They are on page one of that PDF, in text, four hundred bytes
in.

Twenty Drive files are linked from the site. Twenty are PDFs, and seventeen
carry a real text layer - about 47,000 characters between them. The other
three are scans with no text at all, and there is nothing to be done about
those short of OCR, which is a different project: a wrong answer read out of a
bad OCR pass is worse than the honest refusal they get now.

Deliberately not a general PDF pipeline. No OCR, no layout reconstruction, no
tables. pypdf's text extraction on a text-layer PDF is the whole of it, and
anything it cannot read is skipped rather than guessed at.
"""

import io
import logging
import re

log = logging.getLogger("vrhs.pdfs")

TIMEOUT = 45

# The largest file the site links is 6.5 MB. This is a guard against a
# mis-linked video, not a judgement about documents.
MAX_BYTES = 12 * 1024 * 1024

# A 48-page handbook is in here and worth reading. Past that it is not a
# school document, it is an archive.
MAX_PAGES = 60

# Below this there was no text layer worth having: the three scans return
# nothing at all, and the smallest genuine document is 155 characters.
MIN_CHARS = 120

WORDS_PER_CHUNK = 150

# Ligatures pypdf hands back as single codepoints. Curly quotes are left
# alone - they are correct, and the only reason they look wrong is a console
# that cannot print them.
LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl",
}


def clean(text):
    """Readable text, with the page furniture taken out.

    Line breaks are kept. In prose they are noise and the chunker drops them
    anyway; in a song they are the verse, and "Go you Rangers Fight you
    Rangers Take that ball and score" is a worse answer than the four lines it
    came from.
    """
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)

    # A hyphen at the end of a line is a word split across it.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    lines = [" ".join(line.split()) for line in text.split("\n")]
    out, blanks = [], 0
    for line in lines:
        if line:
            out.append(line)
            blanks = 0
        else:
            # One blank line separates verses and paragraphs. Several are the
            # PDF's margins.
            blanks += 1
            if blanks == 1 and out:
                out.append("")
    while out and not out[-1]:
        out.pop()
    return "\n".join(out)


def extract(data):
    """Text from PDF bytes, or None when there is no text layer."""
    try:
        import pypdf
    except ImportError:
        log.warning("pypdf is not installed; PDFs will not be read")
        return None

    if not data[:5].startswith(b"%PDF"):
        return None

    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # Some are encrypted with an empty owner password, which pypdf
            # will open. The rest are genuinely closed.
            try:
                reader.decrypt("")
            except Exception:
                return None
        pages = reader.pages[:MAX_PAGES]
        text = "\n".join((page.extract_text() or "") for page in pages)
    except Exception as e:
        log.info("could not parse pdf: %s", e)
        return None

    text = clean(text)
    if len(text) < MIN_CHARS:
        return None
    return text


def fetch(url, session=None):
    """Download a Drive file and return its text, or None.

    Streamed against a size cap rather than read whole, because the cap has to
    hold for a file whose Content-Length lies or is absent.
    """
    import requests

    get = (session or requests).get
    try:
        response = get(url, timeout=TIMEOUT, stream=True)
        if response.status_code != 200:
            response.close()
            return None
        chunks, total = [], 0
        for block in response.iter_content(65536):
            chunks.append(block)
            total += len(block)
            if total > MAX_BYTES:
                log.info("pdf over %d bytes, skipped: %s", MAX_BYTES, url[:70])
                response.close()
                return None
        response.close()
    except Exception as e:
        log.info("could not fetch pdf %s: %s", url[:70], e)
        return None

    return extract(b"".join(chunks))


def pieces(text):
    """Chunks that keep their line breaks.

    Accumulated by line rather than sliced by word, so a verse or a numbered
    step is not cut in half at the 150th word and the structure survives into
    the chunk the model is handed.
    """
    out, current, words = [], [], 0
    for line in text.split("\n"):
        n = len(line.split())
        if words + n > WORDS_PER_CHUNK and current:
            out.append("\n".join(current).strip())
            current, words = [], 0
        current.append(line)
        words += n
    if current:
        tail = "\n".join(current).strip()
        if tail:
            out.append(tail)
    return [p for p in out if p]
