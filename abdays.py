"""
Which day of the A/B rotation a date is, worked out from the school's own pages.

"Is tomorrow an A day or B day?" is the question students ask most and the
one the site never answers in words. The A/B calendar is a scanned PDF: twelve
months with the B days shaded. images.py reads its printed text - the
"Important Dates" list comes out exactly - but deliberately not its shading,
because a vision model asked which squares were pink invented a pattern that
was wrong on most days.

The shading is redundant anyway. School days alternate A, B, A, B from the
first day of school; days with no school are skipped; a C day (the PSAT)
runs its own schedule and does not advance the rotation. So the rotation is
computed from what the pages print:

- the calendar's Important Dates give the school year, the first and last
  day, and every day off;
- the announcements name the C days ("Oct. 20 PSAT C Day Schedule").

Nothing here is a date. A new calendar on the site is a new rotation on the
next rebuild. Checked against the 2026-27 calendar's shading by hand: every
pink square is a B day by this rule and every B day is pink, and the AP
exam schedule's own A/B labels agree.

If the calendar cannot be found or parsed, there is no rotation, and the bot
falls back to retrieval and the calendar link rather than to a guess.
"""

import datetime
import re

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MONTH = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"

# "AUG. 12 | First Day of School", "DEC. 21-JAN. 1 | Winter Break",
# "OCT. 12-13 | Student/Staff Break". Separator and case vary between
# transcriptions, so the pattern is loose about both.
IMPORTANT = re.compile(
    r"\b" + _MONTH + r"\s*(\d{1,2})"
    r"(?:\s*[-–—]\s*(?:" + _MONTH + r"\s*)?(\d{1,2}))?"
    r"\s*[|:\-–—]\s*([A-Za-z][^\n|]{2,80})", re.I)
# "Oct. 20 PSAT C Day", "C Day - Oct. 20".
C_DAY = re.compile(
    r"\b" + _MONTH + r"\s*(\d{1,2})\b[^.\n]{0,60}?\bC[\s-]?day\b"
    r"|\bC[\s-]?day\b[^.\n]{0,40}?\b" + _MONTH + r"\s*(\d{1,2})\b", re.I)
YEARS = re.compile(r"\b(20\d\d)\s*[-–/]\s*(20\d\d)\b")

# Labels on the Important Dates list that are still school days.
SCHOOL_DAY = re.compile(r"first day|last day|early release", re.I)
# And ones that are about staff before students arrive, not days off.
STAFF_ONLY = re.compile(r"new teacher", re.I)


def _date(month, day, years):
    m = MONTHS[month[:3].lower()]
    return datetime.date(years[0] if m >= 7 else years[1], m, int(day))


def _span(a, b):
    out, d = [], a
    while d <= b:
        out.append(d)
        d += datetime.timedelta(days=1)
    return out


class Rotation:
    def __init__(self, days, off, early, first, last, c_days, url):
        self.days, self.off, self.early = days, off, early
        self.first, self.last, self.c_days, self.url = first, last, c_days, url

    def describe(self, day):
        """One short sentence about a date, or None outside the school year."""
        if (day < self.first - datetime.timedelta(days=30)
                or day > self.last + datetime.timedelta(days=30)):
            return None
        name = f"{day:%A, %B} {day.day}"
        if day in self.days:
            letter = self.days[day]
            extra = [x for x, hit in (
                ("first day of school", day == self.first),
                ("last day of school", day == self.last),
                ("early release", day in self.early)) if hit]
            tail = f" ({', '.join(extra)})" if extra else ""
            return f"{name} is a{'n' if letter == 'A' else ''} {letter} day{tail}."
        if day in self.off:
            return f"{name}: no school - {self.off[day]}."
        if day.weekday() >= 5:
            return f"{name}: weekend, no school."
        if day < self.first:
            return f"{name}: before the first day of school ({self.first:%B} {self.first.day})."
        return f"{name}: after the last day of school ({self.last:%B} {self.last.day})."

    def next_school_day(self, day):
        d = day + datetime.timedelta(days=1)
        for _ in range(40):
            if d in self.days:
                return d
            d += datetime.timedelta(days=1)
        return None

    def context_line(self, today):
        """The A/B days around today, as a line of context, or ""."""
        tomorrow = today + datetime.timedelta(days=1)
        lines = [self.describe(today), self.describe(tomorrow)]
        nxt = self.next_school_day(today)
        if nxt and nxt != tomorrow:
            letter = self.days[nxt]
            lines.append(f"The next school day, {nxt:%A, %B} {nxt.day}, is "
                         f"a{'n' if letter == 'A' else ''} {letter} day.")
        lines = [l for l in lines if l]
        if not lines:
            return ""
        return ("From the VRHS A/B Academic Calendar "
                f"([A/B Calendar]({self.url})): " + " ".join(lines))


def build(docs):
    """A Rotation from the indexed chunks, or None if the calendar is absent."""
    calendar = [d for d in docs
                if re.search(r"A\s*/?\s*_?B", d.get("label") or "")
                and re.search(r"calendar", d.get("label") or "", re.I)
                and IMPORTANT.search(d.get("text", ""))]
    if not calendar:
        return None
    # The newest year among the calendar's chunks: the site has carried last
    # year's calendar alongside this one before.
    def year_of(d):
        m = YEARS.search((d.get("label") or "") + " " + d["text"])
        return (int(m.group(1)), int(m.group(2))) if m else None
    dated = [(year_of(d), d) for d in calendar if year_of(d)]
    if not dated:
        return None
    years = max(y for y, _ in dated)
    chunks = [d for y, d in dated if y == years]
    text = "\n".join(d["text"] for d in chunks)
    url = chunks[0].get("source", "")

    first = last = None
    off, early = {}, set()
    for m in IMPORTANT.finditer(text):
        m1, d1, m2, d2, label = m.groups()
        label = " ".join(label.split()).rstrip(" .")
        try:
            start = _date(m1, d1, years)
            end = _date(m2 or m1, d2, years) if d2 else start
        except ValueError:
            continue
        if end < start or (end - start).days > 20:
            continue
        if re.search(r"first day", label, re.I):
            first = start
        if re.search(r"last day", label, re.I):
            last = end
        if re.search(r"early release", label, re.I):
            early.update(_span(start, end))
        if SCHOOL_DAY.search(label) or STAFF_ONLY.search(label):
            continue
        for d in _span(start, end):
            off.setdefault(d, label)
    if not first or not last or last <= first:
        return None

    c_days = set()
    for d in docs:
        for m in C_DAY.finditer(d.get("text", "")):
            month, day = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            try:
                c = _date(month, day, years)
            except ValueError:
                continue
            if first <= c <= last:
                c_days.add(c)

    days, nxt = {}, "A"
    for d in _span(first, last):
        if d.weekday() >= 5 or d in off:
            continue
        if d in c_days:
            days[d] = "C"
            continue
        days[d] = nxt
        nxt = "B" if nxt == "A" else "A"
    return Rotation(days, off, early, first, last, c_days, url)
