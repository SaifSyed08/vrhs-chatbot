"""Checks for /site-icon and siteicon.py, which find a site's icon.

It fetches third-party addresses and decodes what they send, so most of
these are about what it refuses: hosts outside the index, private
addresses, images that would unpack too large, and slow or broken servers.
Needs the index and the network (the school's own icon is fetched for real).

    python eval/site_icon_check.py
"""
import http.server
import io
import os
import struct
import sys
import tempfile
import threading
import time
import zlib

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main  # noqa: E402
import siteicon  # noqa: E402

failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + str(detail) if detail else ""))


def png_bytes(size, mode="L"):
    out = io.BytesIO()
    Image.new(mode, size).save(out, "PNG")
    return out.getvalue()


def chunk(kind, body):
    return (struct.pack(">I", len(body)) + kind + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))


def as_ico(frame):
    """A one-frame ICO whose directory says 16x16, whatever the frame holds."""
    return (struct.pack("<HHH", 0, 1, 1)
            + struct.pack("<BBBBHHII", 16, 16, 0, 0, 1, 32, len(frame), 22) + frame)


client = main.app.test_client()

print("== the route ==")
r = client.get("/site-icon/vrhs.leanderisd.org")
size = Image.open(io.BytesIO(r.data)).size if r.status_code == 200 else None
check("the school's own icon is found, by the separate process",
      r.status_code == 200 and size == (64, 64), (r.status_code, size))
check("and found again with www. and capitals",
      client.get("/site-icon/WWW.VRHS.LEANDERISD.ORG.").status_code == 200)
r = client.get("/site-icon/example.com")
check("a host outside the index goes to Google's service",
      r.status_code == 302 and r.headers["Location"].endswith("domain=example.com"),
      (r.status_code, r.headers.get("Location")))

print("\n== addresses ==")
for url, want in [("https://vrhs.leanderisd.org/", True),
                  ("http://127.0.0.1/", False), ("http://localhost:8080/", False),
                  ("http://10.0.0.5/", False), ("http://169.254.169.254/latest", False),
                  ("http://[::1]/", False), ("file:///etc/passwd", False),
                  ("http://papercut/", None), ("https://does-not-exist.invalid/", None)]:
    got = siteicon.public_address(url)
    check("public_address(%s) is %s" % (url, want), got is want, got)

print("\n== images that unpack too large ==")
t = time.monotonic()
check("a 10000x10000 PNG is refused before decoding",
      siteicon.as_png(png_bytes((10000, 10000))) is None and time.monotonic() - t < 2)
big = png_bytes((9400, 9400))
check("an ICO hiding a 9400x9400 PNG behind a 16x16 directory",
      siteicon.declared_too_big(as_ico(big)) and siteicon.as_png(as_ico(big)) is None)
small_ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 0, 0, 0, 0))
two_ihdrs = siteicon.PNG_SIGNATURE + small_ihdr + big[8:]
junk_first = siteicon.PNG_SIGNATURE + chunk(b"tEXt", b"k\x00v") + big[8:]
check("a PNG with a 16x16 IHDR and then a 9400x9400 one, in an ICO",
      siteicon.declared_too_big(as_ico(two_ihdrs)))
check("a PNG with a chunk before its 9400x9400 IHDR, in an ICO",
      siteicon.declared_too_big(as_ico(junk_first)))
small = io.BytesIO()
Image.new("RGBA", (48, 48), (200, 0, 0, 255)).save(small, "ICO", sizes=[(16, 16), (48, 48)])
check("an ordinary ICO still converts", siteicon.as_png(small.getvalue()) is not None)
check("an ordinary large PNG is not flagged",
      not siteicon.declared_too_big(png_bytes((2344, 2344), "RGBA")))
unit = struct.pack(">I", 4) + b"tEXt" + siteicon.PNG_SIGNATURE
packed = siteicon.PNG_SIGNATURE + unit * (2_000_000 // len(unit))
t = time.monotonic()
refused = siteicon.declared_too_big(packed) and siteicon.declared_too_big(as_ico(packed))
check("2 MB of tiny chunks full of PNG signatures is refused, quickly",
      refused and time.monotonic() - t < 0.5, round(time.monotonic() - t, 3))
check("an ICO claiming 65535 frames is refused",
      siteicon.declared_too_big(struct.pack("<HHH", 0, 1, 65535) + b"\x00" * 64))

print("\n== the process is killed at its deadline ==")
stuck = os.path.join(tempfile.mkdtemp(), "stuck.py")
with open(stuck, "w") as f:
    f.write("import time\ntime.sleep(600)\n")
real_script, real_deadline = main._SITEICON, siteicon.DEADLINE
main._SITEICON, siteicon.DEADLINE = stuck, 2
t = time.monotonic()
got = main.fetch_site_icon("anything")
main._SITEICON, siteicon.DEADLINE = real_script, real_deadline
check("a fetch that never returns is killed and counted as failed",
      got == (None, "failed") and time.monotonic() - t < 4, round(time.monotonic() - t, 1))
if sys.platform.startswith("linux"):
    import subprocess
    probe = """
import io
import siteicon
from PIL import Image
siteicon.limit_memory()
b = io.BytesIO()
Image.new('RGBA', (3000, 3000), (9, 9, 9, 255)).save(b, 'PNG')
print('fits' if siteicon.as_png(b.getvalue()) else 'refused')
try:
    Image.new('RGBA', (8000, 8000))
    print('allocated')
except MemoryError:
    print('stopped')
print(open('/proc/self/oom_score_adj').read().strip())
"""
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         cwd=os.path.dirname(main._SITEICON))
    lines = out.stdout.split()
    check("on Linux a 3000x3000 icon decodes under the cap", lines[:1] == ["fits"], out.stdout + out.stderr[-200:])
    check("a 256 MB allocation is stopped by it", lines[1:2] == ["stopped"], lines)
    check("and the kernel would kill the fetch before the worker", lines[2:3] == ["1000"], lines)
else:
    print("  --   the memory cap applies on Linux only (Render); not checked here")


print("\n== what counts as no icon, and what as a failure to retry ==")
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/reset"):
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            self.wfile.write(b"\x89PNG")
            self.wfile.flush()
            self.connection.shutdown(2)
        elif self.path.startswith("/busy"):
            self.send_response(503)
            self.end_headers()
        elif self.path == "/" and HOME_LINKS:
            body = "".join('<link rel="icon" href="%s">' % h for h in HOME_LINKS).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/favicon.ico" and HOME_LINKS:
            body = png_bytes((32, 32))
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/corrupt"):
            body = siteicon.PNG_SIGNATURE + b"\x00" * 200
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


HOME_LINKS = []
server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
local = "127.0.0.1:%d" % server.server_address[1]
real_public, real_service, real_scheme = (siteicon.public_address,
                                          siteicon.FAVICON_SERVICE, siteicon.SCHEME)
# The local server is on a loopback address, so it is let through; a name
# starting www. is treated as not resolving, as 170 of the index's hosts'
# www. names do not. The home page and /favicon.ico answer 404, so the
# outcome turns on Google's service alone.
siteicon.public_address = lambda url: None if "//www." in url else True
siteicon.SCHEME = "http"
label = {siteicon.NONE: "no icon", siteicon.FAILED: "failed", siteicon.FOUND: "found"}
for path, want in [("/missing", siteicon.NONE), ("/corrupt", siteicon.NONE),
                   ("/busy", siteicon.FAILED), ("/reset", siteicon.FAILED)]:
    siteicon.FAVICON_SERVICE = "http://" + local + path + "?"
    png, outcome = siteicon.fetch(local)
    check("Google's service answering %s -> %s, an unresolvable www. aside"
          % (path, label[want]), outcome == want, label.get(outcome, outcome))
# A home page whose icon links are all malformed, and a good /favicon.ico:
# each bad link is that candidate's problem, and the fetch carries on.
HOME_LINKS[:] = ['data:image/png;base64,abc', 'http://' + 'a' * 70 + '.com/x.ico',
                 'http://[bad']
siteicon.public_address = lambda url: (None if "//www." in url else
                                       real_public(url) if "aaaa" in url else True)
siteicon.FAVICON_SERVICE = "http://" + local + "/missing?"
png, outcome = siteicon.fetch(local)
check("malformed icon links (bad data: URI, 70-char label, http://[bad) don't stop /favicon.ico",
      outcome == siteicon.FOUND, label.get(outcome, outcome))
HOME_LINKS[:] = []
siteicon.public_address, siteicon.FAVICON_SERVICE, siteicon.SCHEME = (
    real_public, real_service, real_scheme)
server.shutdown()

print("\n== who waits ==")
calls = []
real_fetch = main.fetch_site_icon


def slow(host):
    calls.append(host)
    time.sleep(main.SITE_ICON_WAIT + 2)
    return None, "failed"


main.fetch_site_icon = slow
spare = sorted(h for h in main.icon_hosts() if h.endswith(".edu"))
host = spare[0]
main._SITE_ICONS.pop(host, None)
t = time.monotonic()
r1 = client.get("/site-icon/" + host)
waited = time.monotonic() - t
r2 = client.get("/site-icon/" + host)
check("a slow site costs a request %ss, not the whole fetch" % main.SITE_ICON_WAIT,
      r1.status_code == 404 and waited < main.SITE_ICON_WAIT + 1
      and r1.headers["Cache-Control"] == "no-store", round(waited, 1))
time.sleep(3)
check("and one fetch per host, however many ask", calls == [host], calls)
check("a failed fetch is retried within minutes, not after a restart",
      main._SITE_ICONS[host][1] == "failed"
      and main._SITE_ICONS[host][2] - time.monotonic() <= main.SITE_ICON_TTL["failed"]
      and client.get("/site-icon/" + host).headers["Cache-Control"] == "public, max-age=600")

png = main._SITE_ICONS["vrhs.leanderisd.org"][0]
main._SITE_ICONS["vrhs.leanderisd.org"] = (png, "found", time.monotonic() - 1)
t = time.monotonic()
r = client.get("/site-icon/vrhs.leanderisd.org")
check("an expired icon is served at once while it is refetched",
      r.status_code == 200 and r.data == png and time.monotonic() - t < 1)
time.sleep(main.SITE_ICON_WAIT + 3)
check("and kept when the refetch fails",
      main._SITE_ICONS["vrhs.leanderisd.org"][0] == png)

threading.Thread(target=client.get, args=("/site-icon/" + spare[1],)).start()
time.sleep(0.5)
t = time.monotonic()
r = client.get("/site-icon/" + spare[2])
check("with another host's fetch ahead, a request does not wait",
      r.status_code == 404 and time.monotonic() - t < 1, round(time.monotonic() - t, 2))
t = time.monotonic()
client.get("/site-icon/" + spare[1])
check("but a request for the running one waits for it, though another is queued",
      time.monotonic() - t > 2, round(time.monotonic() - t, 1))
time.sleep(2 * (main.SITE_ICON_WAIT + 2) + 1)
main.fetch_site_icon = real_fetch

print("\n%s" % ("all checks passed" if not failures else "%d FAILED" % failures))
sys.exit(1 if failures else 0)
