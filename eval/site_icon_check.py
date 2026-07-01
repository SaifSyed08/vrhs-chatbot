"""Checks for /site-icon, the route that finds a site's icon.

It fetches third-party addresses from inside the app, so most of these are
about what it refuses: hosts outside the index, private addresses, images
that would unpack too large, and a slow site holding up a request. Needs the
index and the network (the school's own icon is fetched for real).

    python eval/site_icon_check.py
"""
import io
import os
import sys
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
                  ("http://papercut/", False)]:
    check("public_address(%s) is %s" % (url, want), main.public_address(url) is want)

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

print("\n%s" % ("all checks passed" if not failures else "%d FAILED" % failures))
sys.exit(1 if failures else 0)
