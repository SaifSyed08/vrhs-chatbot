"""
Which day of the A/B rotation a date is.

"Is tomorrow an A day or B day?" is the question students ask most and the
one the site cannot answer in words. The A/B calendar is a one-page PDF on
Drive with no text layer - a picture of twelve months with the B days shaded
pink - so ingest finds its filename and nothing else, and the bot could only
hand over the link. A reader filed exactly that as "wrong link".

The calendar is regular, though. School days alternate A, B, A, B from the
first day of school; days with no school are skipped; and a C day (the PSAT)
is its own schedule and does not advance the rotation. So rather than
transcribe a hundred and eighty shaded squares, this encodes the rule and the
year's exceptions, which are the "Important Dates" printed on the calendar.

Checked against the picture month by month: every pink square from August to
May falls on a day this calls B, and the AP exam schedule on the district AP
site, which labels its own dates, agrees too (May 3 A, May 4 B, ...).

Yearly. A new calendar means new dates below - the first and last day, the
no-school days, the C days. Outside the year it knows, it answers nothing
rather than guessing, and the bot falls back to the calendar link.
"""

import datetime

# VRHS 2026-2027 A/B Academic Calendar
# https://drive.google.com/file/d/1n54NUIykpYtq3R8gz9i2-2K7F_Rokv4w/view
CALENDAR_URL = ("https://drive.google.com/file/d/"
                "1n54NUIykpYtq3R8gz9i2-2K7F_Rokv4w/view")
FIRST_DAY = datetime.date(2026, 8, 12)
LAST_DAY = datetime.date(2027, 5, 27)


def _days(start, end):
    d = start
    while d <= end:
        yield d
        d += datetime.timedelta(days=1)


def _span(a, b):
    return set(_days(a, b))


D = datetime.date
# No school for students: holidays, breaks, professional learning days.
NO_SCHOOL = {
    D(2026, 9, 7): "Labor Day",
    D(2026, 9, 21): "Professional Learning",
    D(2026, 10, 9): "Parent-Teacher Conferences / Professional Learning",
    D(2026, 10, 12): "Student/Staff Break",
    D(2026, 10, 13): "Student/Staff Break",
    D(2026, 11, 2): "Continuous Improvement Conference",
    D(2026, 11, 3): "Continuous Improvement Conference",
    D(2027, 1, 4): "Professional Learning",
    D(2027, 1, 18): "Martin Luther King Jr. Day",
    D(2027, 2, 12): "Professional Learning",
    D(2027, 2, 15): "Student/Staff Break",
    D(2027, 2, 16): "Student/Staff Break",
    D(2027, 3, 26): "Student/Staff Break",
    D(2027, 3, 29): "Professional Learning",
    D(2027, 4, 26): "Professional Learning",
}
for _d in _span(D(2026, 11, 23), D(2026, 11, 27)):
    NO_SCHOOL[_d] = "Fall Break"
for _d in _span(D(2026, 12, 21), D(2027, 1, 1)):
    NO_SCHOOL[_d] = "Winter Break"
for _d in _span(D(2027, 3, 15), D(2027, 3, 19)):
    NO_SCHOOL[_d] = "Spring Break"

# Days on their own schedule, outside the rotation.
C_DAYS = {D(2026, 10, 20): "PSAT day"}

EARLY_RELEASE = {D(2026, 12, 18), D(2027, 5, 27)}


def _rotation():
    """Every school day of the year, mapped to "A", "B" or "C"."""
    out, nxt = {}, "A"
    for d in _days(FIRST_DAY, LAST_DAY):
        if d.weekday() >= 5 or d in NO_SCHOOL:
            continue
        if d in C_DAYS:
            out[d] = "C"
            continue
        out[d] = nxt
        nxt = "B" if nxt == "A" else "A"
    return out


ROTATION = _rotation()


def describe(day):
    """One short sentence about a date, or None outside the known year."""
    if day < FIRST_DAY - datetime.timedelta(days=30) or day > LAST_DAY + datetime.timedelta(days=30):
        return None
    name = f"{day:%A, %B} {day.day}"
    if day in ROTATION:
        letter = ROTATION[day]
        extra = []
        if day in C_DAYS:
            extra.append(C_DAYS[day])
        if day in EARLY_RELEASE:
            extra.append("early release")
        if day == FIRST_DAY:
            extra.append("first day of school")
        if day == LAST_DAY:
            extra.append("last day of school")
        tail = f" ({', '.join(extra)})" if extra else ""
        return f"{name} is a{'n' if letter == 'A' else ''} {letter} day{tail}."
    if day in NO_SCHOOL:
        return f"{name}: no school - {NO_SCHOOL[day]}."
    if day.weekday() >= 5:
        return f"{name}: weekend, no school."
    if day < FIRST_DAY:
        return f"{name}: before the first day of school ({FIRST_DAY:%B} {FIRST_DAY.day})."
    return f"{name}: after the last day of school ({LAST_DAY:%B} {LAST_DAY.day})."


def next_school_day(day):
    d = day + datetime.timedelta(days=1)
    for _ in range(40):
        if d in ROTATION:
            return d
        d += datetime.timedelta(days=1)
    return None


def context_line(today):
    """The A/B days around today, as a line of context, or "" if unknown."""
    lines = [describe(today), describe(today + datetime.timedelta(days=1))]
    nxt = next_school_day(today)
    if nxt and nxt != today + datetime.timedelta(days=1):
        lines.append(f"The next school day, {nxt:%A, %B} {nxt.day}, is "
                     f"a{'n' if ROTATION[nxt] == 'A' else ''} {ROTATION[nxt]} day.")
    lines = [l for l in lines if l]
    if not lines:
        return ""
    return ("From the VRHS 2026-2027 A/B Academic Calendar "
            f"([A/B Calendar]({CALENDAR_URL})): " + " ".join(lines))
