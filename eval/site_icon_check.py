"""Checks for /site-icon, the route that finds a site's icon.

It fetches third-party addresses from inside the app, so most of these are
about what it refuses: hosts outside the index, private addresses, images
that would unpack too large, and a slow site holding up a request. Needs the
index and the network (the school's own icon is fetched for real).

    python eval/site_icon_check.py
"""
import http.server
import io
import os
import socket
import struct
import sys
import threading
import time

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main  # noqa: E402

failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(("  ok   " if ok else "  FAIL ") + name + ("  " + str(detail) if detail else ""))


client = main.app.test_client()

# The school's icon, which Google's service does not have, from the site itself.
r = client.get("/site-icon/vrhs.leanderisd.org")
size = Image.open(io.BytesIO(r.data)).size if r.status_code == 200 else None
check("the school's own icon is found", r.status_code == 200 and size == (64, 64),
      (r.status_code, size))
check("and found again with www. and capitals",
      client.get("/site-icon/WWW.VRHS.LEANDERISD.ORG.").status_code == 200)

# Outside the index: not fetched here, handed to Google from the browser.
r = client.get("/site-icon/example.com")
check("a host outside the index goes to Google's service",
      r.status_code == 302 and r.headers["Location"].endswith("domain=example.com"),
      (r.status_code, r.headers.get("Location")))

# Private and odd addresses are never fetched, wherever they come from.
for url, want in [("https://vrhs.leanderisd.org/", True),
                  ("http://127.0.0.1/", False), ("http://localhost:8080/", False),
                  ("http://10.0.0.5/", False), ("http://169.254.169.254/latest", False),
                  ("http://[::1]/", False), ("file:///etc/passwd", False),
                  ("http://papercut/", None)]:
    check("public_address(%s) is %s" % (url, want), main.public_address(url) is want,
          main.public_address(url))

# An image whose header promises more pixels than the cap is refused unread.
big = io.BytesIO()
Image.new("L", (10000, 10000)).save(big, "PNG")
t = time.monotonic()
out = main._as_png(big.getvalue())
check("a 10000x10000 PNG (%d KB) is refused before decoding" % (len(big.getvalue()) // 1024),
      out is None and time.monotonic() - t < 1, round(time.monotonic() - t, 2))

# A slow site: the request gives up after SITE_ICON_WAIT, the fetch carries on,
# and a second request for the same host does not start another.
calls = []
real = main.fetch_site_icon


def slow(host):
    calls.append(host)
    time.sleep(main.SITE_ICON_WAIT + 2)
    return None, "failed"


main.fetch_site_icon = slow
host = next(h for h in sorted(main.icon_hosts()) if h.endswith(".edu"))
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
main.fetch_site_icon = real

# --- second review: what the first round of fences missed -------------------------

# An ICO whose directory says 16x16 but holds a 9400x9400 PNG. Pillow's ICO
# reader decodes the largest frame while opening, so the size check has to
# come from the bytes.
frame = io.BytesIO()
Image.new("L", (9400, 9400)).save(frame, "PNG")
frame = frame.getvalue()
ico = (struct.pack("<HHH", 0, 1, 1)
       + struct.pack("<BBBBHHII", 16, 16, 0, 0, 1, 32, len(frame), 22) + frame)
check("an ICO hiding a 9400x9400 PNG behind a 16x16 directory is refused unopened",
      main.declared_too_big(ico) and main._as_png(ico) is None)
small = io.BytesIO()
Image.new("RGBA", (48, 48), (200, 0, 0, 255)).save(small, "ICO", sizes=[(16, 16), (48, 48)])
check("an ordinary ICO still converts", main._as_png(small.getvalue()) is not None)

# A local server that misbehaves in the ways a real one can. public_address
# is opened up for it, since it is on 127.0.0.1.
class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/drip":
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            try:
                for _ in range(40):
                    self.wfile.write(b"x")
                    self.wfile.flush()
                    time.sleep(1)
            except OSError:
                pass
        elif self.path == "/reset":
            self.send_response(200)
            self.send_header("Content-Length", "100000")
            self.end_headers()
            self.wfile.write(b"\x89PNG")
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
        elif self.path == "/busy":
            self.send_response(503)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
local = "http://127.0.0.1:%d" % server.server_address[1]
real_public = main.public_address
main.public_address = lambda url: True

t = time.monotonic()
try:
    main._icon_get(local + "/drip", main.SITE_ICON_MAX_BYTES, time.monotonic() + 3)
    dripped = "returned"
except TimeoutError:
    dripped = "deadline"
check("a byte-a-second body is cut off at the deadline",
      dripped == "deadline" and time.monotonic() - t < 3 + 2, (dripped, round(time.monotonic() - t, 1)))

for path in ("/reset", "/busy"):
    real_service = main.FAVICON_SERVICE
    main.FAVICON_SERVICE = local + path + "?"
    try:
        png, outcome = main.fetch_site_icon("does-not-exist.invalid")
    finally:
        main.FAVICON_SERVICE = real_service
    check("%s counts as a failure to retry, not 'no icon'" % path, outcome == "failed", outcome)
main.public_address = real_public
check("a name that does not resolve is a failure, not a refusal",
      main.public_address("https://does-not-exist.invalid/") is None)
server.shutdown()

# An icon past its week is served while it is fetched again, and kept if
# the refetch fails.
host = "vrhs.leanderisd.org"
png, _, _ = main._SITE_ICONS[host]
main._SITE_ICONS[host] = (png, "found", time.monotonic() - 1)
calls.clear()
main.fetch_site_icon = slow
t = time.monotonic()
r = client.get("/site-icon/" + host)
check("an expired icon is served at once while it is refetched",
      r.status_code == 200 and r.data == png and time.monotonic() - t < 1)
time.sleep(main.SITE_ICON_WAIT + 3)
check("and kept when the refetch fails", main._SITE_ICONS[host][0] == png)

# With both fetch threads busy, a new host does not wait behind them.
main._ICON_PENDING.clear()
busy = sorted(h for h in main.icon_hosts() if h.endswith(".org") and h not in main._SITE_ICONS)[:3]
for h in busy[:2]:
    threading.Thread(target=client.get, args=("/site-icon/" + h,)).start()
time.sleep(0.5)
t = time.monotonic()
r = client.get("/site-icon/" + busy[2])
check("with the fetch threads busy a request does not wait",
      r.status_code == 404 and time.monotonic() - t < 1, round(time.monotonic() - t, 2))
time.sleep(main.SITE_ICON_WAIT + 3)
main.fetch_site_icon = real

print("\n%s" % ("all checks passed" if not failures else "%d FAILED" % failures))
sys.exit(1 if failures else 0)
