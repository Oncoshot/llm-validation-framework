"""
========================================================================================================================
`to_sortable_date` – a one-stop normaliser for the *messy* date/time stamps you meet in free-text documents
========================================================================================================================

Goal
----
Turn a fragment of text that *might* contain a date (and maybe a time) into one, **lexically-sortable** string.

Why?
•  Sorting these strings with a plain collator will also sort them chronologically.  
•  A single canonical representation makes indexing / deduplication far simpler.  
•  You can still see if information was missing (month, day, year, time granularity) because we never
   “invent” values – we only include what the source really told us.

Output shape
------------
The function returns the *left-most* valid date it can find in one of the following shapes  
(components to the right disappear when absent):

    YYYY
    YYYY-MM
    YYYY-MM-DD
    YYYY-MM-DD HH
    YYYY-MM-DD HH:MM
    YYYY-MM-DD HH:MM:SS
    ????-MM
    ????-MM-DD
    ????-MM-DD HH…            (unknown year but explicit day/time)

Key rules
---------
1. **No gaps** – we never output seconds without minutes, or minutes without hours, etc.
2. **Unknown year** – if the text clearly has *month + day* but no usable year, we insert `"????"`.
3. **Time is optional** – we only add it when a full calendar date (day present) was captured.
4. **Bare hour is ignored** – if the text just says “23” (with no `:`, no AM/PM and no `hours`
   cue) we assume it is not an intentional time and we drop it.  
   *Explicit* hours like “11 PM”, “23 h”, or “around 07 hours” **are** kept.
5. **Timezone is ignored** – we strip `Z`, `UTC`, `+0800`, etc. but *never* shift the clock.
6. **Earliest wins** – if the string contains several dates, the first valid one (left-most) is returned.
7. **Invalid explicit dates** (e.g. `2021-02-29`) make the whole parse fail and return `None`.

Detection strategy
------------------
The parser moves from *most specific* → *least specific* patterns, gathering every match together
with its starting offset; the first chronologically valid candidate wins:

1. ISO week dates: `YYYY-Www-d`
2. Compact `YYYYMMDD`
3. Full day-precision Y-M-D (numeric and textual variants, DM/YM ambiguity handled by `dayFirst`)
4. Year-Month (two components)
5. Unknown-year Month-Day
6. Unknown-year Month
7. Quarter notation (`Q1 2025`)
8. Year-only

Once the earliest *calendar* date is chosen, the same slice of text is examined for a time stamp:

* `HH:MM:SS(.fff)? (AM|PM)?`
* `HH:MM (AM|PM)?`
* `HH (AM|PM)`
* `HH (h|hr|hrs|hour|hours)` or “around HH hours”

Seconds and sub-seconds are trimmed to `SS`; subseconds are discarded.  
AM/PM is converted to 24-hour clock.  
If only `HH` is present **and it isn’t explicit** (e.g. `23` alone) we drop the time entirely.

Dependencies
------------
Only the Python std-lib (`re`, `datetime`) – no `dateutil` needed.

Non-text input
--------------
`raw` is typed `Any` because callers pass values straight from documents and dataframes.
Anything that is not a `str` returns `None` – including `float("nan")`, which is *truthy* and
would otherwise reach `.strip()` and raise.

A two-digit number is only read as a year when nothing time-like follows it: `“Feb 20, 12”` is
February 2012, while `“Feb 20 13:45”` is `????-02-20 13:45` – the 13 is the hour, and the year
stays unknown rather than being invented (ONC-12551).

Canonical dates at a fixed precision
------------------------------------
`to_sortable_date` returns *whatever precision the source stated*, optionally with a time.
A field that is defined to hold a date — a diagnosis date, a treatment start month — wants
one fixed shape instead, so `to_canonical_date` layers a **mask** on top (`YYYY-MM-DD`,
`YYYY-MM`, `YYYY`) and `is_canonical_date` tests whether a value already is one. See those
functions for the rules; both are std-lib only, like everything else here.
"""

import re
from datetime import date
from typing import Any

# ---------------------------------------------------------------------
# Static tables & regexes (unchanged, plus one new for ordinals)
# ---------------------------------------------------------------------
_MONTHS = {k: v for names, v in [
    (['january', 'jan', 'jan.'], 1),
    (['february', 'feb', 'feb.'], 2),
    (['march', 'mar', 'mar.'], 3),
    (['april', 'apr', 'apr.'], 4),
    (['may'], 5),
    (['june', 'jun', 'jun.'], 6),
    (['july', 'jul', 'jul.'], 7),
    (['august', 'aug', 'aug.'], 8),
    (['september', 'sep', 'sep.', 'sept', 'sept.'], 9),
    (['october', 'oct', 'oct.'], 10),
    (['november', 'nov', 'nov.'], 11),
    (['december', 'dec', 'dec.'], 12),
] for k in names}

_ORDINAL_RE = re.compile(r'\b(\d{1,2})(st|nd|rd|th)\b', re.IGNORECASE)


# ---------------------------------------------------------------------
# Utilities (existing + minor tweaks)
# ---------------------------------------------------------------------
def _month_from_name(tok):
    return _MONTHS.get(tok.lower())


def _normalize_two_digit_year(ystr):
    y = int(ystr)
    return 2000 + y if y < 100 else y


def _valid_ym(y, m):
    return 1000 <= y <= 9999 and 1 <= m <= 12


def _valid_ymd(y, m, d):
    try:
        date(y, m, d)
        return True
    except Exception:
        return False


def _valid_unknown_ymd(m, d):
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return False
    if m in (4, 6, 9, 11) and d > 30:
        return False
    if m == 2 and d > 29:
        return False
    return True


def _earliest(matches):
    """Return (start, string, end) for the earliest non‑None match list."""
    return min((x for x in matches if x is not None), key=lambda t: t[0]) if matches else None


# ---------------------------------------------------------------------
# New helper – time extraction
# ---------------------------------------------------------------------
_TIME_RE = re.compile(r"""
    ^[\sT\-,.(]*            # leading junk, spaces, 'T', punctuation
    (?:(?:at|around)\s+)?   # optional 'at' / 'around'
    (?P<h>\d{1,2})          # hour
    (?:
        :(?P<m>\d{1,2})     # minutes
        (?:
            :(?P<s>\d{1,2}) # seconds
            (?:\.\d+)?      #   + fractional
        )?
    )?
    \s*
    (?P<ampm>[AaPp][Mm])?   # AM/PM (not any two of A/P/M — 'PP' is not a time)
    (?:\s*(?P<unit>h|hrs?|hours?)\b)?       # explicit hour cue: 23 h / 07 hours
    (?:\s*(?:Z|UTC|\(UTC\)|[+-]\d{2}:?\d{2}))?  # trailing tz we ignore
""", re.VERBOSE)

# What makes a bare two-digit number a *time* rather than a year: an immediately following
# `:MM`, an AM/PM qualifier, or an hour unit. The year patterns consult this so that
# "Feb 20 13:45" is not read as February 2013.
_TIME_CONTEXT_RE = re.compile(r"""
    ^(?: :\d{1,2}                  # 13:45
       | \s*[APap][Mm]\b           # 11 PM
       | \s*(?:h|hrs?|hours?)\b    # 23 h / 07 hours
    )
""", re.VERBOSE)


def _year_is_actually_time(ystr, text, end):
    """True when a *two-digit* year candidate is really the hour of a following time."""
    return len(ystr) == 2 and _TIME_CONTEXT_RE.match(text[end:]) is not None


def _extract_time(substring):
    """
    Parse a time immediately following the date fragment.
    Returns a normalised string (HH, HH:MM or HH:MM:SS) or None.
    """
    m = _TIME_RE.match(substring)
    if not m:
        return None

    h = int(m.group('h'))
    mnt = m.group('m')
    sec = m.group('s')
    ampm = m.group('ampm')
    unit = m.group('unit')

    # Bare hour is ignored: only AM/PM or an explicit unit ("23 h", "around 07 hours")
    # marks a lone number as an intentional time.
    if not ampm and not unit and mnt is None:
        return None

    # Impossible clock values are not a time: emitting them would break the one guarantee
    # this function makes, that the result is a valid lexically-sortable stamp. Before the
    # check, "29:75:90" came back verbatim and "23:30 PM" became "35:30".
    if (mnt is not None and int(mnt) > 59) or (sec is not None and int(sec) > 59):
        return None
    if ampm and not 1 <= h <= 12:
        return None

    # 12‑hour conversion
    if ampm:
        if ampm.lower() == 'pm' and h != 12:
            h += 12
        if ampm.lower() == 'am' and h == 12:
            h = 0

    if not 0 <= h <= 23:
        return None

    # Midnight '00:00' with no AM/PM => ignore. The hour test matters: without it *every*
    # whole-hour stamp was dropped, so "13:00" lost its time.
    if not ampm and not unit and h == 0 and mnt == '00' and sec is None:
        return None

    if mnt is None:
        return f"{h:02d}"
    if sec is None:
        return f"{h:02d}:{int(mnt):02d}"
    return f"{h:02d}:{int(mnt):02d}:{int(sec):02d}"


# ---------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------
def to_sortable_date(raw: Any, dayFirst: bool = True) -> str | None:
    """
    Convert *raw* to a sortable date (optionally with time) as described
    in the doc‑string of the original implementation.
    """
    # `raw` is Any because callers pass straight from documents/dataframes. Anything that is
    # not text has no date in it — and note float("nan") is *truthy*, so a NaN cell would
    # otherwise reach .strip() and raise.
    if not isinstance(raw, str) or not raw.strip():
        return None

    s = re.sub(r'\s+', ' ', raw.strip())
    s_clean = _ORDINAL_RE.sub(r'\1', s)

    # Holders for candidates at different precisions
    ymd, ym, uymd, um = [], [], [], []
    found_invalid_ymd = found_invalid_ym = False

    # --- 1. ISO week date -------------------------------------------------
    m = re.search(r'\b(\d{4})-W(\d{2})-(\d)\b', s_clean)
    if m:
        try:
            d = date.fromisocalendar(int(m[1]), int(m[2]), int(m[3]))
            date_str = f"{d.year:04d}-{d.month:02d}-{d.day:02d}"
            end_idx = m.end()
            time = _extract_time(s_clean[end_idx:])
            return f"{date_str} {time}" if time else date_str
        except Exception:
            return None

    # --- 2. Compact YYYYMMDD (e.g. 20210405) ------------------------------
    for m in re.finditer(r'\b(1[5-9]\d{2}|20\d{2}|21\d{2})(\d{2})(\d{2})\b', s_clean):
        y, mm, dd = int(m[1]), int(m[2]), int(m[3])
        if _valid_ymd(y, mm, dd):
            ymd.append((m.start(), f"{y:04d}-{mm:02d}-{dd:02d}", m.end()))
        else:
            found_invalid_ymd = True

    # --- 3. Full Y‑M‑D in various flavours --------------------------------
    # Numeric YYYY‑MM‑DD / YYYY/MM/DD / YYYY.MM.DD
    # The boundary is a lookahead, not a consumed character: a date ending in prose
    # punctuation ("…on 2021-07-04.") must still match at day precision, and an invalid
    # one ("2021-02-29,") must fail the parse rather than silently degrade to 2021-02.
    for pat in (r'\b(\d{4})-(\d{1,2})-(\d{1,2})(?=T|\s|[.,;:)\]}]|$)',
                r'\b(\d{4})/(\d{1,2})/(\d{1,2})\b',
                r'\b(\d{4})\.(\d{1,2})\.(\d{1,2})\b'):
        for m in re.finditer(pat, s_clean):
            y, mm, dd = int(m[1]), int(m[2]), int(m[3])
            if _valid_ymd(y, mm, dd):
                ymd.append((m.start(), f"{y:04d}-{mm:02d}-{dd:02d}", m.end()))
            else:
                found_invalid_ymd = True

    # Ambiguous slash dates (D/M/Y or M/D/Y)
    for m in re.finditer(r'\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})\b', s_clean):
        # A 2-digit "year" that is really an hour belongs to the time, not the date.
        if _year_is_actually_time(m[3], s_clean, m.end()):
            continue
        p1, p2, py = int(m[1]), int(m[2]), m[3]
        y = _normalize_two_digit_year(py.replace("'", ""))
        d, mm = (p1, p2) if dayFirst else (p2, p1)
        if _valid_ymd(y, mm, d):
            ymd.append((m.start(), f"{y:04d}-{mm:02d}-{d:02d}", m.end()))
        else:
            found_invalid_ymd = True

    # Monthname Day Year (“Feb 20, 2012”)
    for m in re.finditer(r"\b([A-Za-z]{3,9}\.?)[ ]+(\d{1,2})(?:,)?[ ]+(?:'\s*)?(\d{2,4})\b", s_clean):
        mon = _month_from_name(m[1])
        if mon:
            # A 2-digit "year" that is really an hour belongs to the time, not the date.
            if _year_is_actually_time(m[3], s_clean, m.end()):
                continue
            d   = int(m[2])
            y   = _normalize_two_digit_year(m[3].replace("'", ""))
            if _valid_ymd(y, mon, d):
                ymd.append((m.start(), f"{y:04d}-{mon:02d}-{d:02d}", m.end()))
            else:
                found_invalid_ymd = True

    # Day Month Year (“5 Jan 2020”)
    for m in re.finditer(r"\b(\d{1,2})\s+(?:of\s+)?([A-Za-z]{3,9}\.?),?\s+(?:'\s*)?(\d{2,4})\b", s_clean, re.IGNORECASE):
        mon = _month_from_name(m[2])
        if mon:
            # A 2-digit "year" that is really an hour belongs to the time, not the date.
            if _year_is_actually_time(m[3], s_clean, m.end()):
                continue
            d = int(m[1])
            y = _normalize_two_digit_year(m[3].replace("'", ""))
            if _valid_ymd(y, mon, d):
                ymd.append((m.start(), f"{y:04d}-{mon:02d}-{d:02d}", m.end()))
            else:
                found_invalid_ymd = True

    # Month‑Day‑Year (“Oct‑31‑2021”)
    for m in re.finditer(r'\b([A-Za-z]{3,9}\.?)[-/](\d{1,2})[-/](\d{2,4})\b', s_clean):
        mon = _month_from_name(m[1])
        if mon:
            # A 2-digit "year" that is really an hour belongs to the time, not the date.
            if _year_is_actually_time(m[3], s_clean, m.end()):
                continue
            d = int(m[2])
            y = _normalize_two_digit_year(m[3].replace("'", ""))
            if _valid_ymd(y, mon, d):
                ymd.append((m.start(), f"{y:04d}-{mon:02d}-{d:02d}", m.end()))
            else:
                found_invalid_ymd = True

    # Day‑Month‑Year (“31‑Oct‑2021”)
    for m in re.finditer(r'\b(\d{1,2})[-/]([A-Za-z]{3,9}\.?)[-/](\d{2,4})\b', s_clean):
        mon = _month_from_name(m[2])
        if mon:
            # A 2-digit "year" that is really an hour belongs to the time, not the date.
            if _year_is_actually_time(m[3], s_clean, m.end()):
                continue
            d = int(m[1])
            y = _normalize_two_digit_year(m[3].replace("'", ""))
            if _valid_ymd(y, mon, d):
                ymd.append((m.start(), f"{y:04d}-{mon:02d}-{d:02d}", m.end()))
            else:
                found_invalid_ymd = True

    if found_invalid_ymd:
        return None
    if ymd:
        start, date_str, end_idx = _earliest(ymd)
        time = _extract_time(s_clean[end_idx:])
        return f"{date_str} {time}" if time else date_str

    # --- 4. Year‑Month ----------------------------------------------------
    # Numeric YYYY‑MM / YYYY.MM / YYYY/MM
    for pat in (r'\b(\d{4})-(\d{1,2})\b', r'\b(\d{4})\.(\d{1,2})\b', r'\b(\d{4})/(\d{1,2})\b'):
        for m in re.finditer(pat, s_clean):
            y, mm = int(m[1]), int(m[2])
            if _valid_ym(y, mm):
                ym.append((m.start(), f"{y:04d}-{mm:02d}", m.end()))
            else:
                found_invalid_ym = True

    # Month/Year (“05/2021”)
    for m in re.finditer(r'\b(\d{1,2})/(\d{4})\b', s_clean):
        mm, y = int(m[1]), int(m[2])
        if _valid_ym(y, mm):
            ym.append((m.start(), f"{y:04d}-{mm:02d}", m.end()))
        else:
            found_invalid_ym = True

    # Monthname Year (“March 2021”)
    for m in re.finditer(r'\b([A-Za-z]{3,9}\.?)[ ]+(\d{4})\b', s_clean):
        mon = _month_from_name(m[1])
        if mon:
            y = int(m[2])
            if _valid_ym(y, mon):
                ym.append((m.start(), f"{y:04d}-{mon:02d}", m.end()))
            else:
                found_invalid_ym = True
        
    # Monthname + 4-digit year with dash or slash (e.g., Jan-2024, Jan/2024)
    for m in re.finditer(r'\b([A-Za-z]{3,9}\.?)\s*[-/]\s*(\d{4})\b', s_clean):
        mon = _month_from_name(m[1])
        if not mon:
            continue
        y = int(m[2])
        if _valid_ym(y, mon):
            ym.append((m.start(), f"{y:04d}-{mon:02d}", m.end()))
        else:
            found_invalid_ym = True

    # Monthname + 2-digit year with apostrophe, dash, or slash (e.g. Jan'24, Jan '24, Jan/24, Jan-24)
    for m in re.finditer(r"\b([A-Za-z]{3,9}\.?)\s*(?:[\'’]|[-/])\s*(\d{2})\b", s_clean):
        mon = _month_from_name(m[1])
        if not mon:
            continue
        y = _normalize_two_digit_year(m[2])
        if _valid_ym(y, mon):
            ym.append((m.start(), f"{y:04d}-{mon:02d}", m.end()))
        else:
            found_invalid_ym = True

    if found_invalid_ym:
        return None
    if ym:
        return _earliest(ym)[1]   # never attach time to YM

    # --- 5. Unknown‑year Month‑Day ---------------------------------------
    # Month name + day (space, dash, or slash)
    for m in re.finditer(r'\b([A-Za-z]{3,9}\.?)\s*[-/ ]\s*(\d{1,2})\b', s_clean):
        mon = _month_from_name(m[1])
        if mon:
            d = int(m[2])
            if _valid_unknown_ymd(mon, d):
                uymd.append((m.start(), f"????-{mon:02d}-{d:02d}", m.end()))

    # Day + Month name (space "of", or dash/slash)
    for m in re.finditer(r'\b(\d{1,2})\s+(?:of\s+)?([A-Za-z]{3,9}\.?)\b', s_clean, re.IGNORECASE):
        mon = _month_from_name(m[2])
        if mon:
            d = int(m[1])
            if _valid_unknown_ymd(mon, d):
                uymd.append((m.start(), f"????-{mon:02d}-{d:02d}", m.end()))

    for m in re.finditer(r'\b(\d{1,2})\s*[-/ ]\s*([A-Za-z]{3,9}\.?)\b', s_clean):
        mon = _month_from_name(m[2])
        if mon:
            d = int(m[1])
            if _valid_unknown_ymd(mon, d):
                uymd.append((m.start(), f"????-{mon:02d}-{d:02d}", m.end()))

    if uymd:
        start, date_str, end_idx = _earliest(uymd)
        time = _extract_time(s_clean[end_idx:])
        return f"{date_str} {time}" if time else date_str

    # --- 6. Unknown‑year Month‑only --------------------------------------
    um = [(m.start(), f"????-{_month_from_name(m[1]):02d}", m.end())
          for m in re.finditer(r'\b([A-Za-z]{3,9}\.?)+\b', s_clean)
          if _month_from_name(m[1])]
    if um:
        return _earliest(um)[1]

    # --- 7. Quarter collapse --------------------------------------------
    m = re.search(r'\bQ([1-4])\s+(\d{4})\b', s_clean, re.IGNORECASE)
    if m:
        y = int(m[2])
        if 1000 <= y <= 9999:
            return f"{y:04d}"

    # --- 8. Year‑only -----------------------------------------------------
    m = re.search(r'\b(1[5-9]\d{2}|20\d{2}|21\d{2})\b', s_clean)
    if m:
        return f"{int(m[1]):04d}"

    return None


# ---------------------------------------------------------------------
# Canonical dates: one fixed precision, for a field defined to hold a date
# ---------------------------------------------------------------------
# A mask, mapped to how many components of `YYYY-MM-DD` it keeps.
DATE_MASKS = {"YYYY-MM-DD": 3, "YYYY-MM": 2, "YYYY": 1}

# The shapes a canonical date can take: a real year, optionally a month, optionally a day.
# Zero-padded, `-`-separated, nothing else — no time, no unknown-year `????`.
_CANONICAL_DATE_RE = re.compile(r"^\d{4}(-\d{2}){0,2}$")


def _precision(mask: str) -> int:
    """How many components `mask` keeps. Raises `ValueError` for a mask that isn't one."""
    try:
        return DATE_MASKS[mask]
    except KeyError:
        raise ValueError(
            f"unknown date mask {mask!r}; expected one of {list(DATE_MASKS)}"
        ) from None


def to_canonical_date(raw: Any, mask: str = "YYYY-MM-DD", dayFirst: bool = True) -> str | None:
    """`raw`'s date at `mask`'s precision, or **None** when it holds no usable date.

    Built on `to_sortable_date`, with three rules on top:

    1. **A date, not a timestamp.** Any time component is dropped — a date field stores a
       date.
    2. **The mask truncates; it never pads.** `"26/11/2024"` under `YYYY-MM` is
       `"2024-11"`. A source *coarser* than the mask stays coarse: `"Nov 2024"` under
       `YYYY-MM-DD` is `"2024-11"`, never `"2024-11-01"` — this function does not invent a
       component the source did not state, so a caller can tell "the source said November"
       from "the source said 1 November".
    3. **No year, no date.** `to_sortable_date` reports a year-less date as `"????-11-26"`;
       here that is None, because an unknown-year value cannot be compared with, or read
       back as, a real date.

    None is also what an unparsable value, an impossible calendar date (`"2024-02-30"`) and
    a non-string give back. It is deliberately **not** a sentinel: `"-"`, `""` and `None`
    mean different things to different callers (no information vs. not labelled vs. absent),
    so mapping None onto one of them is the caller's decision — see `llmvalidate.cells`.

    `dayFirst` selects the reading of an ambiguous numeric date, exactly as in
    `to_sortable_date`: `"05/01/2023"` is 5 January when True and 1 May when False. Raises
    `ValueError` for a mask that isn't one of `DATE_MASKS`.
    """
    components = _precision(mask)
    sortable = to_sortable_date(raw, dayFirst=dayFirst)
    if not sortable:
        return None
    day_part = sortable.split(" ", 1)[0]          # rule 1: drop any time
    if not day_part[:4].isdigit():                # rule 3: `????-...` has no year
        return None
    return "-".join(day_part.split("-")[:components])         # rule 2: truncate only


def is_canonical_date(value: Any, mask: str = "YYYY-MM-DD", dayFirst: bool = True) -> bool:
    """True when `value` already *is* a canonical date for `mask` — at that precision or coarser.

    A test of shape, not of correctness: `"2024-11"` passes for a `YYYY-MM-DD` mask (the
    source stated only a month), while `"2024-11-26"` fails for a `YYYY-MM` mask (finer
    than the field holds). Anything `to_canonical_date` would have rewritten fails —
    `"26/11/2024"`, `"2024-11-26 09:30"`, `"2024-1-5"`, `"2024-02-30"`, `"????-11-26"` —
    which makes `to_canonical_date` idempotent on everything this accepts.

    Sentinels are the caller's business: `""` and `"-"` are not canonical dates and return
    False here. Pair this with `llmvalidate.cells.is_no_finding` / `is_unlabelled` to decide
    what a whole cell may hold. An unknown `mask` raises `ValueError` as it does in
    `to_canonical_date`, and is checked before anything else — so a bad argument is reported
    whatever value it was handed, not only for the values that reach the round-trip.
    """
    _precision(mask)
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not _CANONICAL_DATE_RE.match(stripped):
        return False
    # Round-trip: only a value this module would itself have produced is canonical. This is
    # what rejects impossible calendar dates, which the shape check alone lets through.
    return to_canonical_date(stripped, mask, dayFirst) == stripped
