"""
Read the two feeds the front page embeds, rather than only framing them.

The school's home page is mostly two windows onto somewhere else. One is a
Google Calendar iframe showing seven district and campus calendars; the other
is a ParentSquare widget carrying the weekly "Vista Ridge Insider" and the
one-off notices between them. Both are the freshest information the school
publishes, and both were invisible to the crawl for the same reason: the page
contains a frame, and the content is on another origin.

What that cost is easy to state. Picture day, homecoming, the AP registration
deadline, every student holiday and every board meeting are in those two
frames and nowhere else on the site. Asked about any of them the bot said it
could not find anything - correctly, on the corpus it had.

Nothing here is keyed to a particular calendar or a particular school. Both
are found by URL shape in the page markup, the same way gdocs finds an
embedded spreadsheet, so a calendar added to a page next year is read on the
next ingest without anyone editing a list.

Two shapes of thing, chunked differently, for the same reason gdocs splits a
sheet by row and a document by word count - the right unit depends on what it
is:

  * A calendar becomes one chunk per month. One chunk per event floods
    retrieval with fifty near-identical short strings that all match "when is
    the next holiday" about equally, and the answer to that question is
    usually a list anyway.

  * A ParentSquare post becomes word windows like any prose, because an
    Insider is a newsletter and its sections are paragraphs.

Everything fails soft. A feed that will not load leaves the rest of the ingest
untouched: these are other people's servers, and the corpus was built without
them until now.
"""

import base64
import binascii
import datetime
import html
import logging
import re
import urllib.parse

log = logging.getLogger("vrhs.feeds")

# The iframe on the home page. Its query string carries one base64 src= per
# calendar, which is how seven calendars arrive in one embed.
CALENDAR_EMBED = re.compile(
    r"calendar\.google\.com/calendar/embed\?[^\s\"'<>\\]+")

# Either URL identifies the school; the widget is the one that renders posts,
# and it is derivable from the id, so a page linking only /feeds still works.
PARENTSQUARE = re.compile(
    r"parentsquare\.com/schools/(\d+)/(?:rss_widget|feeds)")

TIMEOUT = 45

# How much of a calendar is worth carrying. A week back, because "was there
# school on Monday" is a real question the day after; eight months forward,
# which reaches the end of the school year from any point in the autumn
# without pulling in next year's provisional dates.
WINDOW_BACK = datetime.timedelta(days=7)
WINDOW_AHEAD = datetime.timedelta(days=240)

# A month with more entries than this is split, so one chunk stays inside the
# size a retrieved chunk actually gets to use.
EVENTS_PER_CHUNK = 22

# A bounded daily recurrence is a multi-day event written the way Google
# writes one - band camp, cheer tryouts, a run of holidays. Expanding those is
# worth doing and is unambiguous. Anything else keeps its single occurrence and
# says in words how it repeats, because a half-implemented RRULE that silently
# invents dates is worse than one that does not try.
MAX_EXPANSION = 30

WORDS_PER_CHUNK = 150


# === Discovery =============================================================

def calendar_ids(markup):
    """Every Google Calendar id embedded in a page, in the order they appear."""
    out = []
    for match in CALENDAR_EMBED.finditer(markup or ""):
        query = urllib.parse.urlparse(
            "https://" + html.unescape(match.group(0))).query
        for encoded in urllib.parse.parse_qs(query).get("src", []):
            # Google strips the padding out of the URL, so put it back before
            # decoding rather than letting the decode raise.
            padded = encoded + "=" * (-len(encoded) % 4)
            try:
                cid = base64.b64decode(padded).decode("utf-8")
            except (binascii.Error, UnicodeDecodeError, ValueError):
                continue
            if "@" in cid and cid not in out:
                out.append(cid)
    return out


def parentsquare_ids(markup):
    """Every ParentSquare school id referenced by a page."""
    out = []
    for match in PARENTSQUARE.finditer(markup or ""):
        if match.group(1) not in out:
            out.append(match.group(1))
    return out


# === iCalendar =============================================================

def unfold(text):
    """iCalendar's continuation lines, joined back into whole properties.

    A long SUMMARY is wrapped at 75 octets and continued on a line starting
    with a space. Reading the file line by line without this gives you half an
    event title and no sign that it was ever longer.
    """
    return text.replace("\r\n", "\n").replace("\n ", "").replace("\n\t", "")


def unescape_text(value):
    """iCalendar's backslash escapes, undone."""
    out, i = [], 0
    while i < len(value):
        ch = value[i]
        if ch == "\\" and i + 1 < len(value):
            nxt = value[i + 1]
            out.append({"n": "\n", "N": "\n"}.get(nxt, nxt))
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def properties(block):
    """The block's properties as {NAME: (params, value)}, last one winning."""
    found = {}
    for line in block.split("\n"):
        if ":" not in line:
            continue
        head, _, value = line.partition(":")
        name, _, params = head.partition(";")
        found[name.strip().upper()] = (params, unescape_text(value.strip()))
    return found


def load_zone(name):
    """The calendar's timezone, or None if this machine has no database.

    tzdata is in requirements.txt precisely so this resolves: Linux ships a
    system zone database and Windows does not, so without it the module works
    in production and silently degrades on the machine it is developed on -
    the worst of the two orders to find out in.
    """
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name or "America/Chicago")
    except Exception as e:
        log.warning("no timezone database for %s (%s); times will be "
                    "omitted rather than stated wrongly", name, e)
        return None


def parse_stamp(params, value):
    """(date, (hour, minute) or None, in_utc) from a DTSTART/DTEND.

    The third element is the part that took a wrong answer to notice. Google
    writes timed events as UTC with a trailing Z - "DTSTART:20260917T231500Z"
    - and reading that as a local wall clock turned a board meeting at 6:15 PM
    into one at 11:15 PM. For anything after 6 PM Central it moved the day as
    well: the November meeting came out on the 20th, and it is on the 19th.
    All-day events carry no Z and no time, and must not be shifted at all -
    a holiday is a date, not an instant.
    """
    value = (value or "").strip()
    if re.fullmatch(r"\d{8}", value):
        try:
            return datetime.datetime.strptime(value, "%Y%m%d").date(), None, False
        except ValueError:
            return None, None, False

    match = re.fullmatch(r"(\d{8})T(\d{2})(\d{2})(\d{2})(Z?)", value)
    if not match:
        return None, None, False
    try:
        day = datetime.datetime.strptime(match.group(1), "%Y%m%d").date()
    except ValueError:
        return None, None, False
    if "VALUE=DATE" in (params or ""):
        return day, None, False
    return (day, (int(match.group(2)), int(match.group(3))),
            match.group(5) == "Z")


def to_local(day, clock, in_utc, zone):
    """A stamp moved into the calendar's own timezone.

    Returns (date, clock) with the clock dropped if it cannot be placed - a
    date with no time is incomplete, a date with the wrong time is false, and
    only one of those sends somebody to a meeting five hours late.
    """
    if clock is None or not in_utc:
        return day, clock
    if zone is None:
        return day, None
    moment = datetime.datetime(day.year, day.month, day.day, clock[0],
                               clock[1], tzinfo=datetime.timezone.utc)
    local = moment.astimezone(zone)
    return local.date(), (local.hour, local.minute)


def rule_parts(value):
    parts = {}
    for piece in (value or "").split(";"):
        key, _, val = piece.partition("=")
        if key:
            parts[key.strip().upper()] = val.strip()
    return parts


DAY_NAMES = {"MO": "Monday", "TU": "Tuesday", "WE": "Wednesday",
             "TH": "Thursday", "FR": "Friday", "SA": "Saturday",
             "SU": "Sunday"}


def describe_rule(parts):
    """A recurrence in words, for the rules this module will not expand."""
    freq = parts.get("FREQ", "").lower()
    every = {"daily": "every day", "weekly": "every week",
             "monthly": "every month", "yearly": "every year"}.get(freq, "")
    if not every:
        return ""

    days = parts.get("BYDAY", "")
    if days:
        spelled = [DAY_NAMES.get(d[-2:], "") for d in days.split(",")]
        spelled = [d for d in spelled if d]
        if spelled:
            every = "every week on " + ", ".join(spelled)

    until = parse_stamp("", parts.get("UNTIL", ""))[0]
    if until:
        return "%s until %s %s" % (every, until.day,
                                   until.strftime("%B %Y"))
    return every


def occurrences(start, rrule):
    """The days an event actually falls on, and how it repeats in words.

    Only a bounded FREQ=DAILY is expanded, and that is the whole of the
    judgement here. Everything else in these calendars that recurs is either
    historical or described well enough by its first date plus a sentence, and
    guessing at monthly-by-weekday arithmetic to cover a case that does not
    arise would be inventing dates for the model to state as fact.
    """
    if not rrule:
        return [start], ""

    parts = rule_parts(rrule)
    if parts.get("FREQ") != "DAILY":
        return [start], describe_rule(parts)

    try:
        step = max(1, int(parts.get("INTERVAL") or 1))
    except ValueError:
        step = 1

    until = parse_stamp("", parts.get("UNTIL", ""))[0]
    try:
        count = int(parts["COUNT"]) if "COUNT" in parts else None
    except ValueError:
        count = None

    # An unbounded daily rule would run forever. One occurrence and a
    # description is the honest reading of it.
    if not until and not count:
        return [start], describe_rule(parts)

    days, day = [], start
    while len(days) < MAX_EXPANSION:
        if until and day > until:
            break
        days.append(day)
        if count and len(days) >= count:
            break
        day = day + datetime.timedelta(days=step)
    return days, ""


def calendar_events(cal_id, today, session=None):
    """(calendar name, [event]) for one public Google Calendar.

    Public calendars publish iCalendar at a fixed path with no key and no
    consent screen, which is the only reason this is possible at all - the
    embed itself is a rendered page with the data behind script.
    """
    import requests

    url = ("https://calendar.google.com/calendar/ical/%s/public/basic.ics"
           % urllib.parse.quote(cal_id, safe=""))
    get = (session or requests).get
    try:
        response = get(url, timeout=TIMEOUT)
    except Exception as e:
        log.warning("could not read calendar %s: %s", cal_id[:28], e)
        return None, []
    if response.status_code != 200:
        log.info("calendar %s is not public (%s)", cal_id[:28],
                 response.status_code)
        return None, []

    text = unfold(response.text)
    name_match = re.search(r"X-WR-CALNAME:(.*)", text)
    name = (name_match.group(1).strip() if name_match else "School calendar")
    zone_match = re.search(r"X-WR-TIMEZONE:(.*)", text)
    zone = load_zone(zone_match.group(1).strip() if zone_match else None)

    first = today - WINDOW_BACK
    last = today + WINDOW_AHEAD
    events = []

    for block in text.split("BEGIN:VEVENT")[1:]:
        block = block.split("END:VEVENT")[0]
        props = properties(block)
        if "DTSTART" not in props:
            continue
        # A cancelled occurrence still ships in the feed.
        if props.get("STATUS", ("", ""))[1].upper() == "CANCELLED":
            continue

        summary = " ".join(props.get("SUMMARY", ("", ""))[1].split())
        if not summary:
            continue

        start, clock, in_utc = parse_stamp(*props["DTSTART"])
        if not start:
            continue
        start, clock = to_local(start, clock, in_utc, zone)

        days, repeats = occurrences(start, props.get("RRULE", ("", ""))[1])
        location = " ".join(props.get("LOCATION", ("", ""))[1].split())

        for day in days:
            if first <= day <= last:
                events.append({"day": day, "time": clock, "summary": summary,
                               "location": location, "repeats": repeats})

    events.sort(key=lambda e: (e["day"], e["time"] or (0, 0), e["summary"]))
    return name, events


def clock_text(clock):
    hour, minute = clock
    suffix = "AM" if hour < 12 else "PM"
    return "%d:%02d %s" % (hour % 12 or 12, minute, suffix)


def event_line(event):
    # "%B %d" zero-pads the day on every platform; %-d is not portable to
    # Windows, where this is developed.
    when = event["day"].strftime("%A, %B ") + str(event["day"].day)
    if event["time"]:
        when += " at " + clock_text(event["time"])
    line = when + ": " + event["summary"]
    if event["location"]:
        line += " (%s)" % event["location"]
    if event["repeats"]:
        line += " - repeats %s" % event["repeats"]
    return line + "."


def calendar_chunks(name, events, source):
    """One chunk per month, split again if a month is unusually full."""
    months, order = {}, []
    for event in events:
        key = (event["day"].year, event["day"].month)
        if key not in months:
            months[key] = []
            order.append(key)
        months[key].append(event)

    out = []
    for key in order:
        group = months[key]
        heading = datetime.date(key[0], key[1], 1).strftime("%B %Y")
        for i in range(0, len(group), EVENTS_PER_CHUNK):
            batch = group[i:i + EVENTS_PER_CHUNK]
            part = ("" if len(group) <= EVENTS_PER_CHUNK
                    else " (part %d)" % (i // EVENTS_PER_CHUNK + 1))
            out.append("Events on the %s for %s%s: %s"
                       % (name, heading, part,
                          " ".join(event_line(e) for e in batch)))

    return [{"text": t, "source": source, "kind": "feed", "label": name}
            for t in out]


# === ParentSquare ==========================================================

def parentsquare_posts(school_id, session=None):
    """(title, permalink, posted, text) for each post in a school's widget.

    The widget is a rendered page rather than the RSS its name suggests, so
    this reads the rendered posts. That is exactly the feed the front page
    shows, and it is public - the widget exists to be embedded on school
    websites.
    """
    import requests
    from bs4 import BeautifulSoup

    url = "https://www.parentsquare.com/schools/%s/rss_widget" % school_id
    get = (session or requests).get
    try:
        response = get(url, timeout=TIMEOUT)
    except Exception as e:
        log.warning("could not read ParentSquare %s: %s", school_id, e)
        return []
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    posts = []

    for item in soup.select(".rss-widget-feed-list-item"):
        link = item.select_one(".rss-widget-feed-content-title-link")
        title = " ".join(
            (link.get_text(" ", strip=True) if link else "").split())
        href = (link.get("href") if link else "") or url
        body = item.select_one(".rss-widget-feed-content-description")
        when = item.select_one(".rss-widget-feed-content-date")

        # The widget serves post bodies HTML-escaped inside the page, so what
        # get_text returns still carries &nbsp; and &amp; as literal text.
        # Unescaping after extraction is what turns those back into characters
        # rather than embedding the entity names.
        text = html.unescape(body.get_text(" ", strip=True)) if body else ""
        text = " ".join(text.replace("\xa0", " ").split())
        if not title or len(text) < 40:
            continue

        posted = " ".join(
            when.get_text(" ", strip=True).split()) if when else ""
        posts.append((title, href, posted, text))

    return posts


def parentsquare_chunks(school_id, session=None):
    out = []
    for title, href, posted, text in parentsquare_posts(school_id, session):
        head = ("From the school's ParentSquare announcements, \"%s\"%s"
                % (title, (", posted %s" % posted) if posted else ""))
        words = text.split()
        for i in range(0, len(words), WORDS_PER_CHUNK):
            piece = " ".join(words[i:i + WORDS_PER_CHUNK])
            if piece:
                out.append({"text": "%s: %s" % (head, piece),
                            "source": href,
                            "kind": "feed",
                            "label": title})
    return out


# === Entry point ===========================================================

def read_feeds(pages, today=None):
    """Chunks for every calendar and announcement feed embedded in `pages`.

    `pages` is (url, markup) - the raw markup rather than the parsed soup,
    because the home page's calendar iframe is written by script after load
    and its src never becomes an attribute. Same reason page_embeds scans raw
    markup for Google file URLs.
    """
    import requests

    today = today or datetime.date.today()
    session = requests.Session()
    chunks = []
    seen_calendars, seen_schools = set(), set()

    try:
        for source, markup in pages:
            for cal_id in calendar_ids(markup):
                if cal_id in seen_calendars:
                    continue
                seen_calendars.add(cal_id)
                name, events = calendar_events(cal_id, today, session)
                if not events:
                    continue
                made = calendar_chunks(name, events, source)
                chunks.extend(made)
                log.info("calendar %s: %d events in window, %d chunks",
                         name[:44], len(events), len(made))

            for school in parentsquare_ids(markup):
                if school in seen_schools:
                    continue
                seen_schools.add(school)
                made = parentsquare_chunks(school, session)
                chunks.extend(made)
                log.info("ParentSquare school %s: %d chunks", school,
                         len(made))
    finally:
        session.close()

    log.info("feeds: %d chunks from %d calendars and %d announcement feeds",
             len(chunks), len(seen_calendars), len(seen_schools))
    return chunks
