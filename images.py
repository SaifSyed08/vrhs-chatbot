"""
Read the text that lives inside pictures.

A whole class of the school's information is published as an image: the AP
exam schedule is one picture of a table on the AP site, the A/B calendar is a
PDF whose only content is a scanned page, and the athletics, counselling and
PTSA sites post flyers and schedules as graphics. Ingest read every one of
them as nothing - pdfs.py found no text layer, and page text stops at the
<img> tag - so the bot answered "not listed" to questions whose answers were
on the page in plain sight. A reader filed exactly that about their AP
Biology exam.

So every content image on a crawled page, and every page of a PDF with no
text layer, is transcribed by a vision model and indexed as text from that
page. Two rules keep it honest:

- Text only. Asked to report which calendar squares were shaded, the model
  produced a confident A/B/C pattern that was wrong on most days; asked for
  the printed text, it copied the AP table and the calendar's "Important
  Dates" exactly. The prompt asks for printed text and forbids reading
  meaning off colour, and abdays.py derives the rotation from the dates
  rather than from the shading.
- Cached by content. A transcription is keyed on the image's SHA-256 and
  kept in data/image_text.json, which is committed, so the fortnightly
  rebuild only pays for images that are new, and an image that changes is
  read again because its hash changes.
"""

import base64
import hashlib
import io
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import config

log = logging.getLogger("vrhs.images")

CACHE_PATH = os.getenv("VRHS_IMAGE_CACHE", "data/image_text.json")
MODEL = config.VISION_MODEL
TIMEOUT = 30
MAX_BYTES = 8 * 1024 * 1024
# Smaller than this is an icon, a bullet or a thumbnail. A flyer or a table
# worth reading is at least this big at the size the site serves it.
MIN_SIDE = 220
MIN_AREA = 160_000
# Three at once, and a rate limit is waited out. Six at once ran into the
# API's per-minute token limit, and a failed call is not cached, so the AP exam
# schedule - the image this module exists for - was simply skipped.
WORKERS = 3
RETRIES = 5

PROMPT = (
    "This image is from a high school's website. Transcribe the printed text "
    "in it as plain text that a search engine could index, and nothing else.\n"
    "- Copy every word, number, date, time, name, email address and phone "
    "number exactly as printed.\n"
    "- A table: one line per row, cells separated by ' | ', header row "
    "first. Keep every row.\n"
    "- Lists and notes: one item per line.\n"
    "- Do not report which cells, squares or days are coloured, shaded, boxed "
    "or highlighted, and do not infer anything from colour - only text that "
    "is printed. A legend may be copied as the text it prints.\n"
    "- Do not describe the design, logos, photographs or people.\n"
    "- If the image has no informational text - a photograph, a logo, a "
    "decorative banner, a single heading - reply with exactly NONE.")

_IMAGE_HOSTS = ("googleusercontent.com", "sites.google.com/sitesv-images")

_cache = None
_lock = threading.Lock()
_client = None


def _openai():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _load():
    global _cache
    if _cache is None:
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                _cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            _cache = {}
    return _cache


def save():
    """Write the cache. Called once at the end of an ingest."""
    with _lock:
        cache = _load()
        os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=0, sort_keys=True, ensure_ascii=False)


def _big_enough(data):
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Exception:
        return False
    return min(w, h) >= MIN_SIDE and w * h >= MIN_AREA


def _as_png(data):
    """PNG bytes for the API, which takes PNG, JPEG, GIF and WebP only."""
    from PIL import Image
    with Image.open(io.BytesIO(data)) as im:
        if im.format in ("PNG", "JPEG", "WEBP"):
            return data, "image/" + im.format.lower()
        out = io.BytesIO()
        im.convert("RGB").save(out, "PNG")
        return out.getvalue(), "image/png"


def transcribe(data, where=""):
    """The printed text in an image, or None. Cached on the image's hash."""
    key = hashlib.sha256(data).hexdigest()
    with _lock:
        hit = _load().get(key)
    if hit is not None:
        return hit.get("text") or None

    text = ""
    if _big_enough(data):
        try:
            payload, mime = _as_png(data)
            reply = None
            for attempt in range(RETRIES):
                try:
                    reply = _openai().chat.completions.create(
                        model=MODEL, temperature=0, max_tokens=2000,
                        messages=[{"role": "user", "content": [
                            {"type": "text", "text": PROMPT},
                            {"type": "image_url", "image_url": {
                                "url": "data:%s;base64,%s" % (
                                    mime, base64.b64encode(payload).decode()),
                                "detail": "high"}}]}])
                    break
                except Exception as e:
                    if "429" not in str(e) or attempt == RETRIES - 1:
                        raise
                    import time
                    time.sleep(2.5 * (attempt + 1))
            text = (reply.choices[0].message.content or "").strip()
        except Exception as e:
            # Not cached: a failed call should be retried next ingest, not
            # remembered as an image with nothing in it.
            log.info("could not transcribe image from %s: %s", where[:70], e)
            return None
        if re.fullmatch(r"\W*NONE\W*", text, re.I):
            text = ""
    with _lock:
        _load()[key] = {"text": text, "model": MODEL if text else None,
                        "where": where[:200]}
    return text or None


def pdf_text(data, where=""):
    """Text of a PDF with no text layer, read from its page images."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        pictures = [img.data for page in reader.pages[:10]
                    for img in page.images]
    except Exception as e:
        log.info("could not open pdf images from %s: %s", where[:70], e)
        return None
    texts = [t for t in (transcribe(p, where) for p in pictures) if t]
    return "\n\n".join(texts) or None


def page_images(soup):
    """Image URLs on a page worth reading: the ones its own site hosts."""
    found = []
    for tag in soup.find_all("img"):
        src = tag.get("src") or tag.get("data-src") or ""
        if not any(h in src for h in _IMAGE_HOSTS):
            continue
        # Ask for a readable size: Google serves these at the width named
        # after "=w", often a thumbnail.
        src = re.sub(r"=w\d+(-h\d+)?[^/]*$", "", src) + "=w1600"
        if src not in found:
            found.append(src)
    return found


_session = None


def _get_session():
    global _session
    if _session is None:
        import requests
        _session = requests.Session()
        _session.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    return _session


def grab(soup):
    """The bytes of a page's content images, fetched now.

    Now, while the page is open, and not after the crawl: Google Sites serve
    images from signed addresses that expire within minutes. The first
    version collected the addresses during the crawl and downloaded them
    once it was over, and 269 of 316 had expired by then - among them the AP
    exam schedule, the image this whole module exists for. Nothing failed
    loudly; the downloads simply came back empty.
    """
    out = []
    for url in page_images(soup):
        data = _download(url, _get_session())
        if data:
            out.append(data)
    return out


def _download(url, session):
    try:
        response = session.get(url, timeout=TIMEOUT, stream=True)
        if response.status_code != 200:
            return None
        data = b""
        for block in response.iter_content(65536):
            data += block
            if len(data) > MAX_BYTES:
                return None
        return data
    except Exception:
        return None


def read_page_images(items):
    """[(image_bytes, page_url, page_title)] -> [(page_url, title, text)].

    Each distinct image once - by content, since the signed addresses differ
    on every load - on the first page it was found on: a banner repeated
    across a site is one image, read once and indexed once.
    """
    seen, todo = set(), []
    for data, page_url, title in items:
        key = hashlib.sha256(data).hexdigest()
        if key not in seen:
            seen.add(key)
            todo.append((data, page_url, title))

    def work(item):
        data, page_url, title = item
        text = transcribe(data, page_url)
        return (page_url, title, text) if text else None

    with ThreadPoolExecutor(WORKERS) as pool:
        out = [r for r in pool.map(work, todo) if r]
    log.info("read %d of %d page images as text", len(out), len(todo))
    return out
