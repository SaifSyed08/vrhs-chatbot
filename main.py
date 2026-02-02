from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from werkzeug.exceptions import HTTPException
from openai import OpenAI, RateLimitError
import httpx
import json
import logging
import numpy as np
from numpy.linalg import norm
import os
import datetime
import threading
import time
import re
import traceback
from collections import OrderedDict, deque
from urllib.parse import parse_qs, unquote, urlparse
import config
import gdocs
import hallucination
import corpus

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s")
log = logging.getLogger("vrhs")

app = Flask(__name__)

# Marks the end of the answer stream; everything after it is a JSON block
# carrying the verification notice and the source pages.
META_SENTINEL = ":::meta"

SYSTEM_PROMPT = (
    "You are an AI chatbot for Vista Ridge High School who helps users with "
    "their inquiries, issues and requests. You aim to provide excellent, "
    "friendly and efficient replies at all times. Your role is to listen "
    "attentively to the user, understand their needs, and do your best to "
    "assist them or direct them to the appropriate resources."
    "\n\n"
    "Answer only from the context provided. That context is scraped from the "
    "Vista Ridge High School website and is the only thing you know about "
    "this school. Do not fill gaps with general knowledge about how high "
    "schools usually work, and do not infer specifics such as times, dates, "
    "fees, room numbers, staff names or requirements that the context does "
    "not state. A detail invented that way is what costs a student a wasted "
    "trip or a missed deadline."
    "\n\n"
    "If the context does not answer the question, say plainly that you do "
    "not have that information, and point the user to the "
    "[Staff Directory](https://vrhs.leanderisd.org/directory) or the front "
    "office. A short honest answer is better than a confident wrong one."
    "\n\n"
    "Only cite links as [label](url) if they appear in the context. Never "
    "construct a URL yourself, even when the address looks predictable. If a "
    "question is unclear, ask a clarifying question. End your replies on a "
    "positive note. Your creators are Saif Syed and Junayd Elhassan, who are "
    "both in the class of 2026."
)

def school_year(today=None):
    """The academic year that contains `today`, as "2026-2027".

    The site carries pages from more than one year at once: the calendar page
    still says 2025-2026 while the navigation already links 26-27 bell
    schedules. Without knowing today's date the model cannot tell which of the
    two is current, so it was presenting last year's calendar as though it were
    this year's. Derived from the date rather than hardcoded, so it stays true
    after the next rollover.
    """
    today = today or datetime.date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start}-{start + 1}"


def dated_prompt():
    """The system prompt with today's date and school year filled in."""
    today = datetime.date.today()
    return SYSTEM_PROMPT + chr(10) + chr(10) + (
        f"Today is {today:%A, %d %B %Y}, so the current school year is "
        f"{school_year(today)}. Pages on this site are not all updated at the "
        f"same time, and some still describe an earlier year. When the context "
        f"names a school year, say which year it refers to instead of "
        f"presenting it as current, and if only an older year is available, "
        f"say that the current one was not found."
    )


# The HTTP client is configured rather than defaulted. httpx expires an idle
# connection after 5 seconds, and this bot is asked a question every few
# minutes, so the default was paying a fresh DNS lookup and TLS handshake on
# almost every request. Measured over a 12 second gap: 281 ms default against
# 203 ms with the connection held open.
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(
        limits=httpx.Limits(max_keepalive_connections=10,
                            max_connections=20,
                            keepalive_expiry=config.HTTP_KEEPALIVE_EXPIRY),
        timeout=httpx.Timeout(config.HTTP_READ_TIMEOUT,
                              connect=config.HTTP_CONNECT_TIMEOUT)))


# === Scrape and embed school content ===
SITE = "https://vrhs.leanderisd.org"

# Pages that are not year-scoped and can be named directly.
SEED_URLS = [
    SITE + "/",
    SITE + "/calendar",
    SITE + "/campus_information/",
    SITE + "/campus_information/hours-owed",
    SITE + "/directory",
    SITE + "/volunteer",
    SITE + "/parent_resources",
    SITE + "/staar-testing-dates",
]

# Hubs whose navigation is read to find the rest.
HUB_URLS = [SITE + "/", SITE + "/campus_information/"]

# Files that are not pages. Following these wastes a request and parses binary
# as HTML.
SKIP_SUFFIXES = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".doc", ".docx",
    ".xls", ".xlsx", ".ppt", ".pptx", ".zip", ".ics", ".mp4", ".mp3",
)

# Bounds on the crawl. The site is about two dozen pages, so these are a
# safety net rather than a limit that bites.
MAX_PAGES = config.MAX_PAGES
MAX_DEPTH = config.MAX_DEPTH


LINK_STOPLIST = {
    "home", "search this site", "skip to main content", "skip to navigation",
    "more", "read more", "back", "next", "previous", "here", "click here",
    "vista ridge high school", "leander isd", "facebook", "twitter", "x",
    "instagram", "youtube", "login", "log in", "menu",
}


YEAR_IN_LABEL = re.compile(r"(19|20)\d\d")


def label_rank(label):
    """How informative a link label is. A year beats length; length breaks ties."""
    return (1 if YEAR_IN_LABEL.search(label) else 0, len(label))


def link_chunks(links):
    """One small chunk per unique link, alongside the prose chunks.

    A page like parent_resources is mostly a list of twenty links. Chunked at
    150 words that becomes one blob covering twenty unrelated topics, and its
    embedding is a blur that matches none of them well: "where can I find bus
    information" ranked that chunk 27th of 44 even though it held the only bus
    URL on the site. Giving each link its own chunk makes the label the thing
    being matched, which is what a "where do I find X" question is actually
    asking for.
    """
    # A URL is often linked more than once on a page under different labels,
    # and taking the first one loses information: the district calendar is
    # linked as both "2025-2026 LISD District Calendar" and "District
    # Calendar". Keeping the bare label left the model unable to tell which
    # school year the file belonged to, so it presented last year's calendar as
    # this year's. Prefer the label that says the most.
    best = {}
    for label, href, source in links:
        key = href.rstrip("/")
        clean = " ".join(label.split())
        if len(clean) < 4 or clean.lower() in LINK_STOPLIST:
            continue
        current = best.get(key)
        if current is None or label_rank(clean) > label_rank(current[0]):
            best[key] = (clean, href, source)

    chunks = []

    for clean, href, source in best.values():
        # The shared "at Vista Ridge High School" suffix is deliberate. It
        # repeats across every link chunk and does lift the median cosine
        # between unrelated chunks from 0.79 to 0.85, so it was tried without.
        # Trimming it cost 6.7 points of top-3 retrieval and raised the
        # out-of-scope scores, while gate separation stayed at 96% either way.
        # The context earns its keep; the background figure was cosmetic.
        chunks.append({
            "text": f"{clean} at Vista Ridge High School: [{clean}]({href})",
            # The page the link points AT, not the page it was found on.
            #
            # These were attributed to wherever the crawler happened to see the
            # anchor, and most of them sit in site-wide navigation, so the
            # attribution was close to arbitrary. Asked where the library
            # website is, the answer cited "Saturday SAT Test" - a page that
            # says nothing about the library and merely carries the same nav
            # bar. The reader is told to check a source that cannot confirm
            # anything.
            #
            # A link chunk is about its target. That is what it says, what it
            # matched on, and what a reader following the pill wants.
            "source": href,
            # The site's own wording for the link, which beats deriving a name
            # from the URL - source_label() would call the library website
            # "Vrhslibrary".
            "label": clean,
            "kind": "link",
            # Kept for diagnostics: a link that turns out to be wrong is easier
            # to chase when you know which page it came from.
            "found_on": source,
        })

    return chunks


def unwrap_redirect(href):
    """Google Sites wraps external links in a redirect. Recover the target."""
    if "google.com/url" in href:
        target = parse_qs(urlparse(href).query).get("q")
        if target:
            return unquote(target[0])
    return href


def absolute(href):
    href = unwrap_redirect((href or "").strip())
    if href.startswith("/"):
        return SITE + href
    return href


def page_links(soup, source):
    """Every usable link on a page, before the anchors are flattened."""
    found = []
    for anchor in soup.find_all("a", href=True):
        label = anchor.get_text(" ", strip=True)
        href = absolute(anchor.get("href"))
        if label and href.startswith("http"):
            found.append((label, href, source))
    return found


def page_markdown(soup):
    """Page text with links inline as [label](url), rather than appended.

    The previous version collected every link into one "Important Links" block
    at the end of the page, which chunking then split away from the prose that
    explained them. The result was a chunk of bare URLs that matched nothing:
    asking for bus information retrieved the paragraph naming "Bus Routes /
    Smart Tag" at rank 1 while the chunk holding the actual URL sat at rank 27,
    so the answer named the page and could not link to it. Inlining keeps a
    link inside the sentence that gives it meaning.
    """
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for anchor in soup.find_all("a"):
        label = anchor.get_text(" ", strip=True)
        href = absolute(anchor.get("href"))
        if label and href.startswith("http"):
            anchor.replace_with(f"[{label}]({href})")
        else:
            anchor.replace_with(label or "")

    return soup.get_text(" ", strip=True)


def fetch(url):
    """Fetch and parse one page.

    `requests` and `bs4` are imported here rather than at module scope. They
    are reachable only from a rebuild, never from answering a question, so a
    cold start should not pay for them. Measured marginal saving is about
    50 ms, not the 525 ms the two cost in isolation - openai and flask already
    pull most of their dependency tree in, so only the top of it was ours to
    remove. Small, but it is free and the coupling was wrong regardless.
    """
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(url, timeout=25)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def normalise(url):
    """Strip the fragment and query so one page is not crawled several times."""
    url = url.split("#")[0].split("?")[0].rstrip("/")
    return url or SITE


def crawl_urls():
    """Every page on the school site, found by following its own links.

    This replaced a list of topic patterns matched against the navigation. A
    whitelist only finds what somebody thought to name: "when do doors open for
    the Saturday SAT" was unanswerable because no pattern matched
    /saturday-sat-test, even though the homepage links it directly. Nine of the
    site's pages were missing for that reason.

    The site is around two dozen pages, so crawling all of it costs about
    twenty seconds and removes the whole class of gap.
    """
    seen = {normalise(SITE)}
    found = []
    queue = deque([(normalise(SITE), 0)])

    while queue and len(found) < MAX_PAGES:
        url, depth = queue.popleft()
        try:
            soup = fetch(url)
        except Exception as e:
            log.warning(f"Could not read {url}: {e}")
            continue

        found.append(url)
        if depth >= MAX_DEPTH:
            continue

        for anchor in soup.find_all("a", href=True):
            link = normalise(absolute(anchor["href"]))
            if not link.startswith(SITE) or link in seen:
                continue
            if link.lower().endswith(SKIP_SUFFIXES):
                continue
            seen.add(link)
            queue.append((link, depth + 1))

    # Seeds are crawled too, in case one is orphaned from the navigation.
    for url in SEED_URLS:
        if normalise(url) not in found and len(found) < MAX_PAGES:
            found.append(normalise(url))

    return found


def scrape_vrhs_pages():
    urls = crawl_urls()
    log.info(f"crawled {len(urls)} pages from the live site")
    chunks = []

    all_links = []

    for url in urls:
        log.info(f"Scraping {url}...")
        try:
            soup = fetch(url)
            all_links.extend(page_links(soup, url))
            combined_text = page_markdown(soup)

            words = combined_text.split()
            for i in range(0, len(words), 150):
                chunk_text = " ".join(words[i:i + 150])
                if chunk_text:
                    chunks.append({"text": chunk_text, "source": url})
        except Exception as e:
            log.warning(f"Failed to scrape {url}: {e}")

    # Two files on this site are both labelled "A/B Calendar" and one of them
    # is last year's. Nothing in the page text distinguishes them, so the model
    # chose between them blind and chose wrong. Their Drive filenames say
    # exactly which is which - "2025-2026 District A_B Calendar" against "VRHS
    # 2026-2027 Calendar" - so the label carries the filename from here on and
    # the existing preference for year-bearing labels sorts the rest out.
    all_links = gdocs.enrich_labels(all_links)

    link_only = link_chunks(all_links)
    log.info(f"added {len(link_only)} link chunks from {len(all_links)} anchors")
    chunks.extend(link_only)

    # Read the Google Docs the site links to, rather than only pointing at
    # them. Bell times, the senior checklist and the resource one-pagers all
    # live one hop off the site in a doc the crawl walked past, because the
    # crawl is same-domain. 16 of the 23 linked docs export as plain text with
    # no credentials, and answering from what a document says beats handing
    # over its URL.
    for label, url, text in gdocs.linked_documents(all_links):
        words = text.split()
        for i in range(0, len(words), 150):
            piece = " ".join(words[i:i + 150])
            if piece:
                chunks.append({
                    "text": f"From the linked document {label}: {piece}",
                    "source": url,
                    "kind": "document",
                    # The document's own title. source_label() names a page
                    # from its last URL segment, which is fine for the school
                    # site and useless for a Google Doc - every one of them
                    # ends "/edit?usp=sharing", so the pills all read
                    # "Edit?usp=sharing". The title is right there at ingest.
                    "label": label,
                })

    # A page that yields nothing is a silent hole in the knowledge base: the
    # chatbot will confidently not know things the site actually documents.
    covered = {c["source"] for c in chunks}
    missing = [u for u in urls if u not in covered]
    if missing:
        log.warning(f"WARNING: {len(missing)}/{len(urls)} pages produced no chunks:")
        for url in missing:
            log.info(f"  - {url}")

    return chunks


# Facts the scrape cannot reach: links that live behind menus or on other
# domains. Kept as data rather than repeated code so the ingest stays one loop.
MANUAL_CHUNKS = [
    "The Ranger Time Portal for Vista Ridge High School can be accessed here: "
    "[Ranger Time Portal](https://adv.leanderisd.org/login.aspx?ReturnUrl=%2fDefault.aspx)",

    "The Vista Ridge High School Staff Directory, useful for contact info or "
    "finding who manages what, can be accessed here: "
    "[Staff Directory](https://vrhs.leanderisd.org/directory)",

    "The Hours Owed page for Vista Ridge High School can be accessed here: "
    "[Hours Owed](https://vrhs.leanderisd.org/campus_information/hours-owed)",

    "The Attendance page for Vista Ridge High School can be accessed here: "
    "[Attendance](https://sites.google.com/leanderisd.org/vrhsattendance/)",

    "The packet to request a new club/organization can be accessed here: "
    "[new club/organization request packet]"
    "(https://docs.google.com/document/d/1-gTgT9VXpYRzu282hvCj2gJBO5Up-6YIllXjH3F2RZ4/copy)",
]


def build_chunks(clean=True):
    """Scrape, optionally run the hygiene pass, and add the manual chunks.

    `clean=False` reproduces the pre-hygiene corpus for the ablation in
    eval/rebuild_corpora.py.
    """
    chunks = scrape_vrhs_pages()
    if clean:
        # Strip cross-page navigation before embedding. Measured at ~72% of all
        # scraped words; leaving it in is what drove unrelated chunks to a 0.88
        # median cosine and made absolute similarity gating impossible.
        chunks = corpus.prepare(chunks)
    chunks += [{"text": t, "source": "manual"} for t in MANUAL_CHUNKS]
    return chunks


@app.route("/embed")
def embed_chunks():
    chunks = build_chunks()

    for chunk, vector in zip(chunks, embed_texts([c["text"] for c in chunks])):
        chunk["embedding"] = vector

    os.makedirs("data", exist_ok=True)
    with open("data/vrhs_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    return f"Embeddings generated and saved ({len(chunks)} chunks)."


def is_out_of_credit(error):
    """True when a 429 means the account has no quota, not that it is busy.

    OpenAI raises RateLimitError for both. One is transient and worth a retry,
    the other will fail identically forever, and retrying it just makes the
    reader wait longer for the same failure.
    """
    text = str(error).lower()
    return "insufficient_quota" in text or "exceeded your current quota" in text


def with_retry(call, attempts=3, base_delay=0.6):
    """Retry a transient rate limit with backoff. Quota errors raise at once."""
    delay = base_delay
    for attempt in range(attempts):
        try:
            return call()
        except RateLimitError as error:
            if is_out_of_credit(error) or attempt == attempts - 1:
                raise
            log.warning(f"[retry] rate limited, waiting {delay:.1f}s")
            time.sleep(delay)
            delay *= 2.5


def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


# Embedding one question is a network round trip that sits directly in front
# of the answer: nothing can be retrieved, so no chat call can start, until it
# comes back. Measured at a 218 ms median, it is roughly a fifth of the wait
# before any text appears.
#
# Swapping the model does not help - text-embedding-3-small measures 211 ms,
# inside the noise, because the cost is the round trip and not the model. What
# does help is not making the call. A school chatbot is asked the same few
# things over and over ("bell schedule", "when does school start"), so repeat
# questions can answer from a cache and skip the hop entirely.
#
# Keyed on the question with case and whitespace normalised away, which is
# what makes "Bell schedule?" and "bell  schedule?" one entry. Bounded, so a
# long-lived process cannot grow without limit; ada-002 vectors are about
# 12 KB each, so 512 entries is roughly 6 MB.
QUERY_CACHE_SIZE = config.QUERY_CACHE_SIZE
_query_cache = OrderedDict()
_query_cache_lock = threading.Lock()
_query_cache_stats = {"hits": 0, "misses": 0}


def cache_key(text):
    return " ".join(text.lower().split())


def embed_query(text):
    """Embed a user question, reusing a recent identical one when possible."""
    key = cache_key(text)

    with _query_cache_lock:
        if key in _query_cache:
            _query_cache.move_to_end(key)
            _query_cache_stats["hits"] += 1
            return _query_cache[key]
        _query_cache_stats["misses"] += 1

    vector = embed_text(text)

    with _query_cache_lock:
        _query_cache[key] = vector
        _query_cache.move_to_end(key)
        while len(_query_cache) > QUERY_CACHE_SIZE:
            _query_cache.popitem(last=False)

    return vector


def embed_text(text):
    response = with_retry(lambda: client.embeddings.create(
        model=config.EMBED_MODEL, input=text))
    return response.data[0].embedding


# The embeddings endpoint takes a list. Rebuilding one chunk per request meant
# a few hundred sequential round trips, which is slow enough that a hosted
# /embed can hit the platform request timeout before finishing.
EMBED_BATCH = 96


def embed_texts(texts):
    """Embed many strings, batched, preserving input order."""
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        response = with_retry(lambda: client.embeddings.create(
            model=config.EMBED_MODEL, input=batch))
        vectors.extend(item.embedding for item in
                       sorted(response.data, key=lambda d: d.index))
        log.info(f"embedded {min(start + EMBED_BATCH, len(texts))}/{len(texts)}")
    return vectors


EMBEDDINGS_PATH = config.EMBEDDINGS_PATH

# Retrieval quotas, see get_relevant_context. Three prose chunks carry the
# explanation, two link chunks carry somewhere to go.
PROSE_SLOTS = config.PROSE_SLOTS
LINK_SLOTS = config.LINK_SLOTS

# The index was re-read and re-parsed from disk on every question. It is small
# and immutable between rebuilds, so it is loaded once and kept in memory.
_INDEX = {"docs": None, "matrix": None}
_INDEX_LOCK = threading.Lock()


def load_index():
    """Load the corpus once, returning (docs, unit-normalised matrix).

    Locked because warm_start() loads on a background thread while a request
    may already be asking for it. Without the lock both would parse the same
    9.7 MB of JSON; the result is identical either way, so this is about not
    doing the work twice rather than about correctness.
    """
    if _INDEX["docs"] is not None:
        return _INDEX["docs"], _INDEX["matrix"]

    with _INDEX_LOCK:
        if _INDEX["docs"] is None:
            if not os.path.exists(EMBEDDINGS_PATH):
                return None, None
            with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
                docs = json.load(f)
            matrix = np.array([d["embedding"] for d in docs], dtype=np.float64)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
            _INDEX["docs"], _INDEX["matrix"] = docs, matrix

    return _INDEX["docs"], _INDEX["matrix"]


# === Pre-generated answers ===
#
# The two API calls in front of an answer have a floor that no amount of tuning
# reaches: 208 ms to embed the question and 451 ms before gpt-4.1 emits a
# token, both of them round trips whose cost lives on OpenAI's side. Measured
# from a browser that is about 120 ms from Render, a novel question cannot
# realistically be answered in much under 700 ms.
#
# The way under that is to have answered already. A school chatbot is asked the
# same few dozen things all year - bell schedules, absences, hours owed - so
# those answers can be generated ahead of time, checked properly while nobody
# is waiting, and served from memory.
#
# This is the one place the grounding checks stop being advisory. Everywhere
# else they annotate an answer the reader has already seen, because
# verification cannot finish until the last token exists. A pre-generated
# answer has no reader yet, so a variant that fails a check is simply not
# admitted to the cache. prewarm.py does that filtering; nothing that failed
# ever reaches this file.
ANSWER_CACHE_PATH = os.getenv("VRHS_ANSWER_CACHE", "data/answer_cache.json")

_ANSWERS = {"fingerprint": None, "entries": {}, "generated_at": None,
            "model": None}
_ANSWER_TURN = {}
_ANSWER_LOCK = threading.Lock()
_ANSWER_STATS = {"hits": 0, "misses": 0}


def index_fingerprint(docs):
    """Identify the corpus, so a rebuilt index retires answers built on it.

    Hashed over the chunk text rather than the file, because the file carries
    1536 floats per chunk and the mtime changes on every checkout. What matters
    is whether the words an answer was grounded in are still the words in the
    index.
    """
    import hashlib
    h = hashlib.sha256()
    for d in docs:
        h.update(d["text"].encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def load_answer_cache():
    """Load pre-generated answers, or nothing if they do not match the index."""
    if not os.path.exists(ANSWER_CACHE_PATH):
        log.info("[answers] no pre-generated cache at %s", ANSWER_CACHE_PATH)
        return

    try:
        with open(ANSWER_CACHE_PATH, "r", encoding="utf-8") as f:
            blob = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("[answers] could not read cache: %s", e)
        return

    docs, _ = load_index()
    if docs is None:
        return

    current = index_fingerprint(docs)
    if blob.get("index_fingerprint") != current:
        # Refusing to serve is the whole point. An answer generated against an
        # older corpus may cite a page that has since been removed, and it
        # carries a verification verdict that was true of text nobody is
        # retrieving any more.
        log.warning("[answers] cache built for index %s but index is %s - "
                    "ignoring it, re-run prewarm.py",
                    blob.get("index_fingerprint"), current)
        return

    with _ANSWER_LOCK:
        _ANSWERS.update({
            "fingerprint": current,
            "entries": blob.get("entries", {}),
            "generated_at": blob.get("generated_at"),
            "model": blob.get("model"),
        })
    total = sum(len(e.get("variants", []))
                for e in _ANSWERS["entries"].values())
    log.info("[answers] %d questions, %d verified variants, from %s",
             len(_ANSWERS["entries"]), total, _ANSWERS["generated_at"])


def cached_answer(question):
    """A pre-generated answer for this question, or None.

    Variants rotate rather than repeating one phrasing forever. They are not
    alternative facts: prewarm.py admits a set only when every variant cites
    the same pages, so what varies is wording.
    """
    key = cache_key(question)
    with _ANSWER_LOCK:
        entry = _ANSWERS["entries"].get(key)
        if not entry or not entry.get("variants"):
            _ANSWER_STATS["misses"] += 1
            return None
        variants = entry["variants"]
        turn = _ANSWER_TURN.get(key, 0)
        _ANSWER_TURN[key] = (turn + 1) % len(variants)
        _ANSWER_STATS["hits"] += 1
        return variants[turn]


def warm_start():
    """Pay the first-question costs at boot instead of at the first question.

    Deployment target is Cloud Run, which scales to zero. A container that has
    just started has done none of the one-time work the answer path needs, and
    all of it used to land on whoever asked first after a quiet period:

      * parsing the 9.7 MB index and normalising it, measured at 137 ms here
        and slower on a smaller cloud instance;
      * DNS and the TLS handshake to the API, which the connection pool then
        reuses for every later question.

    Both run on a daemon thread so opening the port is not delayed, which is
    what Cloud Run watches to decide the container is ready. Failures are
    swallowed on purpose: this is an optimisation, and a warm-up that cannot
    reach the network must not stop the app from booting. The real request
    path calls load_index() itself and will do the work then if this lost the
    race.
    """
    if config.NO_WARM:
        return

    def run():
        try:
            load_index()
            load_answer_cache()
        except Exception as e:
            log.warning(f"[warm] index preload failed: {e}")
        try:
            if os.getenv("OPENAI_API_KEY"):
                client.embeddings.create(
                    model=config.EMBED_MODEL, input="warm")
        except Exception as e:
            log.warning(f"[warm] connection warm-up failed: {e}")
        else:
            log.info("[warm] index and API connection ready")

    threading.Thread(target=run, name="warm-start", daemon=True).start()


# Slugs that should not be title-cased into "Staar" or "Vrhs".
ACRONYMS = {
    "staar": "STAAR", "vrhs": "VRHS", "lisd": "LISD", "ptsa": "PTSA",
    "pta": "PTA", "voe": "VOE", "acc": "ACC", "ap": "AP",
}


def source_label(url):
    """Readable name for a source pill, derived from the page slug."""
    if url == "manual":
        return "VRHS quick links"
    tail = url.rstrip("/").split("/")[-1]

    # The homepage has no path segment to name it by.
    if not tail or "." in tail:
        return "Vista Ridge High School"

    slug = tail.replace("-", " ").replace("_", " ").strip()

    # Drop leading school-year tokens so "26 27 bell schedules" reads as
    # "Bell Schedules".
    words = slug.split()
    while words and words[0].isdigit():
        words.pop(0)

    labelled = [ACRONYMS.get(w.lower(), w.capitalize()) for w in words]
    return " ".join(labelled) or "Vista Ridge High School"


def get_relevant_context(query):
    """Return the top chunks, similarity stats, and the pages they came from."""
    docs, matrix = load_index()
    if docs is None:
        return "No knowledge base available. Please visit /embed first.", {}, []

    q = np.array(embed_query(query), dtype=np.float64)
    q /= np.linalg.norm(q)
    sims = matrix @ q

    # Prose and links are retrieved under separate quotas rather than as one
    # top-3. There are roughly twice as many link chunks as prose chunks, so a
    # plain top-k fills with links: asking who the principal is returned his
    # name as a link label at rank 2, with nothing saying he is the principal,
    # while the paragraph that said so sat at rank 6. The model then correctly
    # declined to answer. Reserving slots keeps a narrative answer and a place
    # to go in the same context.
    ranked = np.argsort(sims)[::-1]
    is_link = [docs[i].get("kind") == "link" for i in range(len(docs))]

    prose = [i for i in ranked if not is_link[i]][:PROSE_SLOTS]
    links = [i for i in ranked if is_link[i]][:LINK_SLOTS]

    # Keep overall similarity order so the strongest match leads the context.
    order = sorted(set(prose) | set(links), key=lambda i: -sims[i])
    context = "\n\n".join(docs[i]["text"] for i in order)

    # The standard-score margin has to be computed here, while similarities to
    # every chunk are still in hand - it cannot be recovered from the top 3.
    spread = float(sims.std())
    stats = {
        "top": float(sims.max()),
        "mean": float(sims.mean()),
        "z": float((sims.max() - sims.mean()) / spread) if spread else None,
        "n": int(sims.size),
    }

    # Pages behind the retrieved chunks, best first, without repeats.
    #
    # Only the ones that actually matched. Retrieval fills its quotas whether
    # or not there are five good chunks to fill them with, so on a narrow
    # question the last slots go to whatever ranked next - and every one of
    # those used to become a source pill. Three pills under an answer that
    # really came from one page is a quiet overstatement of where the answer
    # came from, and it is the kind a reader cannot check.
    #
    # The cutoff is the same weak threshold the gate uses, so a chunk that
    # would not have convinced the gate does not get to name a page either.
    # The best-matching source is always kept: an answer the gate called solid
    # came from somewhere, and showing nothing would be its own kind of wrong.
    sources, seen = [], set()
    for rank, i in enumerate(order):
        url = docs[i]["source"]
        if url == "manual" or url in seen:
            continue
        if rank > 0 and sims[i] < hallucination.SIMILARITY_WEAK:
            continue
        seen.add(url)
        sources.append({"url": url,
                        "label": docs[i].get("label") or source_label(url),
                        "score": round(float(sims[i]), 4)})

    return context, stats, sources


@app.route("/health")
def health():
    """Deployment check.

    Plain /health is cheap and answers "is this box configured". Adding
    ?deep=1 spends one embedding call to answer the question that actually
    matters in a broken deployment: can this process reach the API, and if
    not, with which error. Without it the only way to see the cause is the
    platform log, which is not always to hand.
    """
    docs, _ = load_index()
    key = os.getenv("OPENAI_API_KEY") or ""

    report = {
        "ok": bool(key) and docs is not None,
        "openai_key_present": bool(key),
        "openai_key_tail": key[-4:] if key else None,
        "index_path": EMBEDDINGS_PATH,
        "index_found": docs is not None,
        "chunks": len(docs) if docs else 0,
        "sources": len({d["source"] for d in docs}) if docs else 0,
        "working_directory": os.getcwd(),
        "data_writable": os.access("data", os.W_OK) if os.path.isdir("data")
                         else None,
        # Hit rate is the thing to watch: it is what says whether skipping the
        # embedding round trip is actually buying anything in production, as
        # opposed to on the repeated questions of a benchmark.
        "query_cache": {
            "entries": len(_query_cache),
            "hits": _query_cache_stats["hits"],
            "misses": _query_cache_stats["misses"],
        },
        # Hit rate here is the number to watch. Pre-generation is only worth
        # its complexity if real questions actually land on the pre-written
        # ones, and a hit rate near zero means the question list was guessed
        # rather than drawn from what students ask.
        "answer_cache": {
            "questions": len(_ANSWERS["entries"]),
            "variants": sum(len(e.get("variants", []))
                            for e in _ANSWERS["entries"].values()),
            "fingerprint": _ANSWERS["fingerprint"],
            "generated_at": _ANSWERS["generated_at"],
            "model": _ANSWERS["model"],
            "hits": _ANSWER_STATS["hits"],
            "misses": _ANSWER_STATS["misses"],
        },
    }

    if request.args.get("deep"):
        try:
            client.embeddings.create(model=config.EMBED_MODEL,
                                     input="health check")
            report["api_call"] = "ok"
        except Exception as e:
            report["ok"] = False
            report["api_call"] = "failed"
            report["api_error_type"] = type(e).__name__
            report["api_error"] = str(e)[:300]

    return jsonify(report)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/report", methods=["POST"])
def submit_report():
    data = request.get_json()
    description = data.get("description")

    if not os.path.exists("data"):
        os.makedirs("data")

    report = {
        "description": description,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    reports_file = "data/reports.json"
    try:
        with open(reports_file, "r") as f:
            reports = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        reports = []

    reports.append(report)

    with open(reports_file, "w") as f:
        json.dump(reports, f, indent=2)

    return jsonify({"status": "success"})


# How much of the conversation goes back to the model. Three exchanges is
# enough for "who is he" to resolve and short enough that the prompt does not
# grow without limit across a long session.
HISTORY_TURNS = 6
HISTORY_CHARS = 1200

# Words that make a question depend on what came before it. A follow-up like
# "who is he" retrieves nothing on its own - it carries none of the words that
# would match a chunk - so the question that gets embedded has to include what
# "he" referred to.
PRONOUN = re.compile(
    r"\b(he|she|they|it|him|her|them|his|hers|their|theirs|its|"
    r"this|that|these|those|there|one)\b", re.I)


def clean_history(raw):
    """The last few turns, trimmed, in the shape the API expects."""
    if not isinstance(raw, list):
        return []

    turns = []
    for item in raw[-HISTORY_TURNS:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = (item.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            turns.append({"role": role, "content": content[:HISTORY_CHARS]})
    return turns


def retrieval_query(question, history):
    """What to embed, which is not always what the user typed.

    "Who is he?" matches nothing in the corpus. It contains no subject, so the
    nearest chunk is whatever happens to be closest to three function words,
    and the answer that follows is grounded in nothing.

    When a question looks like it depends on the one before it, the previous
    user turn is prepended before embedding. Concatenation rather than a model
    call to rewrite it: rewriting sits in front of retrieval, which sits in
    front of the answer, so it would add a round trip to the one path in this
    system that a reader is actually waiting on. Gluing two questions together
    is free and puts the missing subject back in the text, which is all the
    embedding needs.

    Only for questions that look dependent. Concatenating unconditionally would
    blur a genuine change of subject - asking about bell schedules right after
    asking about the principal should retrieve bell schedules.
    """
    if not history:
        return question

    words = question.split()
    dependent = len(words) <= 4 or bool(PRONOUN.search(question))
    if not dependent:
        return question

    previous = next((t["content"] for t in reversed(history)
                     if t["role"] == "user"), None)
    if not previous:
        return question

    log.info("[followup] embedding with prior turn: %s", question[:50])
    return previous + " " + question


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("query")
    history = clean_history(data.get("history"))

    # Checked before retrieval, because a hit skips both round trips - there is
    # no point embedding a question whose answer is already written and already
    # verified. This is the only path that answers without calling the API at
    # all, and it is the only one that can be quick enough to feel instant.
    # Not for a follow-up. The cache is keyed on the question alone and knows
    # nothing about what came before, so serving "who is he" a pre-written
    # answer would answer a different question than the one asked.
    hit = cached_answer(question) if question and not history else None
    if hit:
        def replay():
            # Sent as one chunk. The streaming shape exists so a reader is not
            # staring at nothing while a model writes; there is nothing to wait
            # for here, and pacing it out artificially would be spending the
            # latency this whole path exists to avoid.
            yield hit["answer"]
            yield META_SENTINEL + json.dumps({
                "notice": hit.get("notice", []),
                "sources": hit.get("sources", []),
                "retrieval": hit.get("retrieval", {}),
                "cached": True,
            })

        log.info("[answers] hit  q=%s", question[:60])
        return Response(stream_with_context(replay()),
                        content_type="text/plain; charset=utf-8")

    context, stats, sources = get_relevant_context(
        retrieval_query(question, history))

    # If no KB yet, just send that and stop.
    if context.startswith("No knowledge base"):
        return jsonify({"answer": context})

    # History sits between the system prompt and the current turn, so a
    # follow-up resolves against what was actually said rather than being
    # answered cold. The context block stays attached to the current question
    # rather than being sent as its own turn: it was retrieved for this turn,
    # and pinning it there stops the model treating an earlier turn's context
    # as still in force.
    messages = ([{"role": "system", "content": dated_prompt()}]
                + history
                + [{"role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"}])

    def generate():
        answer = []

        # call the OpenAI API with streaming turned on
        stream = with_retry(lambda: client.chat.completions.create(
            model=config.CHAT_MODEL, messages=messages, stream=True))

        # for each chunk that comes in…
        for chunk in stream:
            # pull out the new text (token) if there is any
            token = chunk.choices[0].delta.content

            if token:
                # keep a copy so the answer can be checked once it is complete
                answer.append(token)
                # yield it straight to the HTTP response
                yield token

        # An answer is only checkable as a whole, so verification runs after
        # the last token. A failed check appends a warning rather than
        # retracting what the user has already read.
        verdict = hallucination.verify_answer(client, "".join(answer), context,
                                              stats, question)
        log.info(f"[grounding] z={verdict.retrieval_z} "
              f"top_sim={verdict.retrieval_top} "
              f"level={verdict.retrieval_level} "
              f"bad_links={len(verdict.bad_links)} "
              f"unsupported={len(verdict.unsupported)}")

        # One trailing JSON block rather than more prose: the client needs the
        # notice and the sources as data, and parsing prose out of a stream was
        # fragile. Sources are withheld when the gate says the corpus does not
        # cover the question, since citing a page that did not support the
        # answer is its own kind of false confidence.
        meta = {
            "notice": verdict.notice_points(),
            "sources": sources if verdict.retrieval_level != "none" else [],
            "retrieval": {
                "level": verdict.retrieval_level,
                "top": verdict.retrieval_top,
            },
        }
        yield META_SENTINEL + json.dumps(meta)

    # Wrap it in Flask’s Response so it streams chunked HTTP
    return Response(
        stream_with_context(generate()),
        content_type='text/plain; charset=utf-8',
    )


@app.errorhandler(Exception)
def handle_unexpected(error):
    """Return the failure as readable text instead of a blank 500 page.

    The stack trace goes to the process log, and the error class goes to the
    reader. A student cannot act on "RateLimitError", but whoever they forward
    the screenshot to can, and it saves a round trip through the logs.
    """
    # Let Flask handle its own 404s and 405s rather than turning them into 500s.
    if isinstance(error, HTTPException):
        return error

    traceback.print_exc()
    log.error(f"[error] {type(error).__name__}: {error}")

    # A student cannot act on "RateLimitError". Say what it means for them, and
    # keep the diagnostic pointer for whoever maintains the deployment.
    if isinstance(error, RateLimitError):
        if is_out_of_credit(error):
            return (
                "The assistant has run out of API credit and cannot answer "
                "right now. Please let a site admin know. (quota exhausted, "
                "see /health?deep=1)",
                503,
            )
        return (
            "The assistant is handling too many questions at once. Please try "
            "again in a few seconds.",
            503,
        )

    return (
        "Something went wrong reaching the assistant "
        f"({type(error).__name__}). If this keeps happening, open /health?deep=1.",
        500,
    )


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """Record a thumbs rating against the retrieval diagnostics for that answer.

    The rating on its own says an answer was poor. Paired with the retrieval
    score it says why: a thumbs-down on a question that scored below the gate
    is missing content, while a thumbs-down on a well-retrieved question is a
    generation or phrasing problem. The two need different fixes.
    """
    data = request.get_json(silent=True) or {}

    entry = {
        "question": (data.get("question") or "")[:500],
        "rating": data.get("rating"),
        "reason": data.get("reason"),
        "retrieval_top": data.get("retrieval_top"),
        "retrieval_level": data.get("retrieval_level"),
        "flagged": bool(data.get("flagged")),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    os.makedirs("data", exist_ok=True)
    path = "data/feedback.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        entries = []

    entries.append(entry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    log.info(f"[feedback] {entry['rating']} level={entry['retrieval_level']} "
          f"q={entry['question'][:60]}")
    return jsonify({"status": "success"})


warm_start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT)
