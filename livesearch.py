"""
When the corpus has nothing, look it up on the district's own sites.

The bot's misses are not random. Over-refusal sits around 10-16%, and most of
it is one shape: the answer exists, but on leanderisd.org rather than on
vrhs.leanderisd.org. The crawl is same-domain by design - it has to be, or it
would wander off into the whole internet - so district-level answers are
permanently out of reach of an index built from the school site.

This is the fallback for exactly that case, and only that case. It runs when
the retrieval gate says nothing in the corpus came close, which is the moment
the alternative is a refusal.

**Why not the model's own web search.** OpenAI's built-in tool is $25 per
thousand calls on a non-reasoning model, which is 2.5 cents every time this
path fires - roughly ten times the cost of the answer it is attached to, on a
bot whose entire monthly spend is a couple of dollars. This costs one
embedding call for a page of titles, about three thousandths of a cent.

**Why the reranking is the whole design.** Both district sites run WordPress,
and its ?s= search is keyword matching with no notion of relevance: asking it
about homecoming returns a story about an elementary school principal, and
asking about graduation returns board briefs from two years ago. Handing those
to the model as "here is what I found" would be worse than saying nothing,
because it would launder noise into an answer that looks sourced. So the
titles are embedded and ranked against the question with the same model the
corpus uses, only the best two pages are fetched, and their text is gated on
cosine before any of it reaches the prompt. Search finds candidates; the
embedding decides.

The gate here is deliberately higher than the corpus gate. The corpus is
scored after the hygiene pass strips cross-page navigation, and these pages
are raw - every one of them carries the district's menu, which drags unrelated
text upward. The two numbers are not comparable, and the safe direction to be
wrong in is silence.

Everything fails soft and everything is bounded. This is someone else's server
on the path of a question a reader is waiting for.
"""

import concurrent.futures
import logging
import re
import time
import urllib.parse

import config

log = logging.getLogger("vrhs.livesearch")

# WordPress search on the two district properties. The school's own site is
# absent on purpose: it is Google Sites, it is already crawled in full, and if
# the answer were there the gate would not have missed it.
SEARCH_URLS = [
    "https://news.leanderisd.org/?s=",
    "https://www.leanderisd.org/?s=",
]

ALLOWED_HOST = re.compile(r"^https://[\w-]+\.leanderisd\.org/")

# Cloudflare sits in front of both and answers a bare python-requests
# User-Agent with a 403. Not a block on automation as such - the same request
# with a browser string is served normally - but it does mean the header is
# load-bearing rather than decorative.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# Both of these are budgets, not limits, and they are set from what the path
# actually cost when it was first measured across sixteen questions: it fired
# on four and added between 3.3 and 7.7 seconds to each, for one answer that
# would otherwise have been a refusal. That is worth paying once; it is not
# worth paying ten seconds for.
#
# So the whole lookup gets seven seconds and each request four, and anything
# that overruns is abandoned. The reader then gets the answer they would have
# got anyway, a little later - never a page that hung.
TIMEOUT = 4
DEADLINE = 7

# Fewer candidates is mostly cheaper embedding, but it is also the search
# results getting worse as they go: page one of a WordPress ?s= is ordered by
# nothing in particular, and by the fourteenth link it is reliably navigation.
MAX_CANDIDATES = 14
MAX_PAGES = 2
MAX_PASSAGES = 3
WORDS_PER_CHUNK = 130

# Titles below this are not worth spending a page fetch on.
TITLE_FLOOR = 0.76
# And a passage below this does not reach the prompt. Set from what the first
# run through it actually returned, which is the only way to pick one of
# these. At 0.80 a question about bad weather days pulled three slices of the
# district's school-safety page, all scoring 0.81 to 0.82, none of them about
# weather - confidently irrelevant text is worse than a refusal, because the
# refusal does not look sourced. The passages that genuinely answered their
# question scored 0.87 and 0.88, so the gap is real and 0.85 sits in it.
#
# See the module note on why this is above the corpus gate rather than equal
# to it.
PASSAGE_FLOOR = 0.85

# At most this many from any one page, so two pages cannot become three
# consecutive paragraphs of one.
PER_PAGE = 2

SKIP_PATHS = ("/category/", "/author/", "/tag/", "/page/", "/feed",
              "/wp-content/", "/wp-admin/")
SKIP_SUFFIX = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".zip",
               ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".mp4")


# Words that cannot narrow a search. A question made only of these has no
# subject to look up, and firing four network requests at it spends five
# seconds to arrive at the same "what do you mean?" the model gives instantly.
STOPWORDS = frozenset("""
a about all am an and any are as at be been being but by can could did do does
doing done for from get give go had has have he her here him his how i if in
is it its know let like me more most my need of off on one or our out over
please said say see she should so some tell than that the their them then
there these they thing things this those to too us use want was we were what
when where which who whom whose why will with would you your
hi hey hello hiya yo sup thanks thank ok okay yeah yep nope please sorry
""".split())


def has_subject(question):
    """Whether there is anything here worth searching for."""
    words = re.findall(r"[a-z0-9']+", (question or "").lower())
    # One is enough. "when is graduation" is a real question with a single
    # content word in it, and requiring two threw it out alongside "what is
    # this" - which is the opposite of the distinction being drawn.
    return len([w for w in words if w not in STOPWORDS]) >= 1


def _session():
    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": UA,
                            "Accept": "text/html,application/xhtml+xml"})
    return session


def candidates(question, session):
    """(title, url) pairs the district search offers for this question."""
    from bs4 import BeautifulSoup

    query = urllib.parse.quote(question[:120])
    found, seen = [], set()

    def one(base):
        try:
            r = session.get(base + query, timeout=TIMEOUT)
            return r.text if r.status_code == 200 else ""
        except Exception as e:
            log.info("search %s failed: %s", base, e)
            return ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pages = list(pool.map(one, SEARCH_URLS))

    for markup in pages:
        if not markup:
            continue
        soup = BeautifulSoup(markup, "html.parser")
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"].split("#")[0].split("?")[0]
            title = " ".join(anchor.get_text(" ", strip=True).split())
            if not ALLOWED_HOST.match(href) or href in seen:
                continue
            if any(p in href for p in SKIP_PATHS):
                continue
            if href.lower().endswith(SKIP_SUFFIX):
                continue
            # A link with no words is a logo or an icon, and a two-word one is
            # navigation. Neither can be ranked against a question.
            if len(title) < 20:
                continue
            seen.add(href)
            found.append((title, href))
            if len(found) >= MAX_CANDIDATES:
                return found
    return found


def page_text(url, session):
    from bs4 import BeautifulSoup
    try:
        r = session.get(url, timeout=TIMEOUT)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
    except Exception as e:
        log.info("could not read %s: %s", url[:70], e)
        return ""

    for tag in soup(["script", "style", "noscript", "nav", "header",
                     "footer", "form"]):
        tag.decompose()
    return " ".join(soup.get_text(" ", strip=True).split())


def _rank(question, texts, embed_texts):
    """Cosine of each text against the question, in one embedding call."""
    import numpy as np

    vectors = embed_texts([question] + list(texts))
    matrix = np.array(vectors, dtype=np.float64)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix[1:] @ matrix[0]


def lookup(question, embed_texts, deadline=DEADLINE):
    """Passages from the district site for a question the corpus missed.

    Returns [] for every failure, which is most of them: no candidates, no
    title close enough to be worth a fetch, nothing on the fetched pages above
    the gate, a timeout, a 403. The caller carries on with the refusal it was
    already going to give.
    """
    if not config.LIVE_SEARCH or not question:
        return []
    if not has_subject(question):
        log.info("[live] no subject to search for in %r", question[:50])
        return []

    started = time.time()
    session = _session()
    try:
        found = candidates(question, session)
        if not found:
            log.info("[live] no candidates for %s", question[:50])
            return []

        if time.time() - started > deadline:
            log.info("[live] out of time after searching")
            return []

        titles = [t for t, _ in found]
        try:
            scores = _rank(question, titles, embed_texts)
        except Exception as e:
            log.warning("[live] could not rank candidates: %s", e)
            return []

        best = sorted(range(len(found)), key=lambda i: -scores[i])
        picked = [(found[i][0], found[i][1], float(scores[i]))
                  for i in best[:MAX_PAGES] if scores[i] >= TITLE_FLOOR]
        if not picked:
            log.info("[live] best title only %.3f for %s",
                     float(max(scores)), question[:50])
            return []

        if time.time() - started > deadline:
            log.info("[live] out of time before fetching")
            return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PAGES) as p:
            bodies = list(p.map(lambda row: page_text(row[1], session),
                                picked))

        pieces, origins = [], []
        for (title, url, _), body in zip(picked, bodies):
            words = body.split()
            for i in range(0, len(words), WORDS_PER_CHUNK):
                piece = " ".join(words[i:i + WORDS_PER_CHUNK])
                if len(piece) > 200:
                    pieces.append(piece)
                    origins.append((title, url))
        if not pieces:
            return []

        if time.time() - started > deadline:
            log.info("[live] out of time before scoring passages")
            return []

        try:
            scores = _rank(question, pieces, embed_texts)
        except Exception as e:
            log.warning("[live] could not score passages: %s", e)
            return []

        order = sorted(range(len(pieces)), key=lambda i: -scores[i])
        out, per_url = [], {}
        for i in order:
            if len(out) >= MAX_PASSAGES or scores[i] < PASSAGE_FLOOR:
                break
            title, url = origins[i]
            if per_url.get(url, 0) >= PER_PAGE:
                continue
            per_url[url] = per_url.get(url, 0) + 1
            out.append({"text": pieces[i], "label": title, "url": url,
                        "score": round(float(scores[i]), 4)})

        log.info("[live] %d passage(s) for %s in %.1fs (best %.3f)",
                 len(out), question[:44], time.time() - started,
                 float(scores[order[0]]) if len(order) else 0.0)
        return out
    finally:
        session.close()
