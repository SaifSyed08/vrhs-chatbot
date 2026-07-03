"""Find a site's icon, as a 64px PNG. Run as its own process by main.py.

    python siteicon.py <host>     PNG on stdout; exit 0 found, 3 none, 4 failed

Google's favicon service knows most sites, but not every one: it has never
picked up the school's own site, whose icon is declared in its
<link rel="icon"> and served from a Google Sites image address. So this asks
Google first and, when Google has nothing, reads the icon the site itself
declares, then /favicon.ico.

Why a process of its own. It fetches third-party addresses and decodes
whatever images they return, and three rounds of review each found a way
past the fences drawn inside the app: a small file that unpacks to hundreds
of megabytes (Pillow's ICO reader decodes the largest frame while opening;
a PNG can carry a second IHDR), and servers that drip bytes in ways that
keep every per-receive timeout happy (a dripped gzip header, dripped
response headers). Inside the app's process, either one could take down the
single worker that streams every answer. Out here the operating system
enforces the limits: the parent kills this process at its deadline,
whatever it is waiting on, and on Linux it cannot allocate more than
MEMORY_LIMIT. The checks below still run, so the usual cases are cheap, but
they are no longer the only line.

The fences that remain the app's to draw:
 - every address, including each redirect and wherever a page's
   <link rel="icon"> points, must resolve to public addresses only;
 - bodies are capped in bytes, and images in pixels from what they declare.
Not fenced: DNS rebinding. The address is resolved to be checked and then
again to be fetched, so a name with a zero TTL could change in between.
What that buys is a blind GET whose body comes back only if it decodes as
an image, from an instance with nothing internal worth reaching.
"""
import base64
import io
import ipaddress
import socket
import struct
import sys
from urllib.parse import urljoin, urlparse

from livesearch import UA

FAVICON_SERVICE = "https://www.google.com/s2/favicons?sz=64&domain="
SIZE = 64
MAX_BYTES = 2_000_000
PAGE_MAX_BYTES = 2_000_000
MAX_PIXELS = 9_000_000          # 3000 x 3000; the school's is 2344 x 2344
FORMATS = ["PNG", "ICO", "GIF", "JPEG", "BMP", "WEBP"]
DEADLINE = 20                   # seconds; the parent kills the process here
MEMORY_LIMIT = 192 * 2 ** 20    # heap, on Linux; about 60 MB at idle
FOUND, NONE, FAILED = 0, 3, 4
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SCHEME = "https"                # the tests point this at a local server


class Transient(Exception):
    """Worth trying again later: a 5xx, a 429, a name that did not resolve."""


def public_address(url):
    """True if the URL is http(s) and its host resolves only to public addresses.

    None if the name did not resolve, which may be a passing failure.
    """
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        return False
    try:
        port = u.port or 443
    except ValueError:
        return False
    try:
        infos = socket.getaddrinfo(u.hostname, port, proto=socket.IPPROTO_TCP)
    except OSError:
        return None
    try:
        return bool(infos) and all(
            ipaddress.ip_address(info[4][0].split("%")[0]).is_global
            for info in infos)
    except ValueError:
        return False


def get(url, limit):
    """(body, final url) for a GET, or (None, url) if refused or not a 200.

    Redirects are followed by hand so each hop is checked to be public.
    Raises Transient, or the network library's own errors, for a failure
    that may pass.
    """
    import requests
    for _ in range(4):
        allowed = public_address(url)
        if allowed is None:
            raise Transient("did not resolve: " + url[:80])
        if not allowed:
            return None, url
        r = requests.get(url, timeout=(4, 6), stream=True,
                         allow_redirects=False, headers={"User-Agent": UA})
        try:
            if r.is_redirect and r.headers.get("location"):
                url = urljoin(url, r.headers["location"])
                continue
            if r.status_code == 429 or r.status_code >= 500:
                raise Transient("%d from %s" % (r.status_code, url[:80]))
            if r.status_code != 200:
                return None, url
            data = r.raw.read(limit + 1, decode_content=True)
            return (data if len(data) <= limit else None), url
        finally:
            r.close()
    return None, url


def _png_sizes(data, start):
    """Every IHDR's (width, height) in the PNG at start, walking its chunks.

    Not just the first: Pillow accepts chunks before IHDR and keeps the last
    IHDR it sees, so a 16x16 header followed by a 9400x9400 one decodes at
    the second size.
    """
    i = start + 8
    while i + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[i:i + 8])
        if kind == b"IHDR":
            if i + 16 > len(data):
                yield (1 << 31, 1 << 31)
                return
            yield struct.unpack(">II", data[i + 8:i + 16])
        if kind == b"IEND":
            return
        i += 12 + length


def declared_too_big(data):
    """True if any image in the file declares more pixels than the cap.

    Every PNG anywhere in the file (which covers an ICO's PNG frames), and
    each BMP frame of an ICO by its own header.
    """
    i = data.find(PNG_SIGNATURE)
    while i != -1:
        for w, h in _png_sizes(data, i):
            if w * h > MAX_PIXELS:
                return True
        i = data.find(PNG_SIGNATURE, i + 1)
    if data[:4] == b"\x00\x00\x01\x00":
        count = struct.unpack("<H", data[4:6])[0]
        for k in range(count):
            entry = 6 + 16 * k
            if len(data) < entry + 16:
                return True
            offset = struct.unpack("<I", data[entry + 12:entry + 16])[0]
            if data[offset:offset + 8] == PNG_SIGNATURE:
                continue
            if len(data) < offset + 12:
                return True
            w, h = struct.unpack("<ii", data[offset + 4:offset + 12])
            if abs(w) * (abs(h) // 2) > MAX_PIXELS:
                return True
    return False


def as_png(data):
    """An icon file in a usual icon format, as a square 64px PNG, or None."""
    from PIL import Image
    if declared_too_big(data):
        return None
    im = Image.open(io.BytesIO(data), formats=FORMATS)
    if im.width * im.height > MAX_PIXELS:
        return None
    im.load()
    im = im.convert("RGBA")
    im.thumbnail((SIZE, SIZE), Image.LANCZOS)
    square = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    square.paste(im, ((SIZE - im.width) // 2, (SIZE - im.height) // 2))
    out = io.BytesIO()
    square.save(out, "PNG", optimize=True)
    return out.getvalue()


def fetch(host):
    """(png or None, outcome): FOUND, NONE (no icon anywhere) or FAILED."""
    import requests
    import urllib3
    from bs4 import BeautifulSoup
    # urllib3's own errors (a read timeout, a reset mid-body) are none of
    # requests' exceptions, so both are named.
    network = (requests.RequestException, urllib3.exceptions.HTTPError,
               Transient, OSError)
    failed = False

    def attempt(url, optional=False):
        # optional: an address guessed rather than given, like www.<host>.
        # That it does not exist says nothing about whether to try again.
        nonlocal failed
        try:
            if url.startswith("data:image") and ";base64," in url:
                data = base64.b64decode(url.split(",", 1)[1], validate=False)
            else:
                data, _ = get(url, MAX_BYTES)
        except network as e:
            failed = failed or not optional
            print("[icon] %s: %s" % (url[:120], e), file=sys.stderr)
            return None
        if not data:
            return None
        try:
            return as_png(data)
        except Exception as e:
            # Not an image, or a broken one: an answer, not a failure.
            print("[icon] %s: %s" % (url[:120], e), file=sys.stderr)
            return None

    got = attempt(FAVICON_SERVICE + host)
    if got:
        return got, FOUND

    base = SCHEME + "://"
    candidates = []
    for home, optional in ((base + host + "/", False),
                           (base + "www." + host + "/", True)):
        try:
            page, final = get(home, PAGE_MAX_BYTES)
        except network as e:
            failed = failed or not optional
            print("[icon] could not read %s: %s" % (home, e), file=sys.stderr)
            continue
        if not page:
            continue
        soup = BeautifulSoup(page, "html.parser")
        for link in soup.find_all("link", href=True):
            rel = " ".join(link.get("rel") or []).lower()
            if "icon" in rel:
                candidates.append((urljoin(final, link["href"].strip()), False))
        candidates.append((urljoin(final, "/favicon.ico"), False))
        break
    # Even with no readable home page, the conventional address may answer.
    candidates += [(base + host + "/favicon.ico", False),
                   (base + "www." + host + "/favicon.ico", True)]
    seen = set()
    for url, optional in candidates:
        if url in seen:
            continue
        seen.add(url)
        got = attempt(url, optional)
        if got:
            return got, FOUND
    return None, (FAILED if failed else NONE)


def limit_memory():
    try:
        import resource
    except ImportError:          # Windows, where this only runs in development
        return
    resource.setrlimit(resource.RLIMIT_DATA, (MEMORY_LIMIT, MEMORY_LIMIT))


if __name__ == "__main__":
    limit_memory()
    try:
        png, outcome = fetch(sys.argv[1])
    except MemoryError:
        # Something unpacked past the limit. It will next time too.
        sys.exit(NONE)
    if png:
        sys.stdout.buffer.write(png)
    sys.exit(outcome)
