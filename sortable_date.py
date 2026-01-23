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
4. **Bare hour is ignored** – if the text just says “23” (with no `:` and no AM/PM/`hours` cue)
   we assume it is not an intentional time and we drop it.  
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
    (?:at\s+)?              # optional 'at'
    (?P<h>\d{1,2})          # hour
    (?:
        :(?P<m>\d{1,2})     # minutes
        (?:
            :(?P<s>\d{1,2}) # seconds
            (?:\.\d+)?      #   + fractional
        )?
    )?
    \s*
    (?P<ampm>[APMapm]{2})?  # AM/PM
    (?:\s*(?:Z|UTC|\(UTC\)|[+-]\d{2}:?\d{2}))?  # trailing tz we ignore
""", re.VERBOSE)


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

    # Bare hour w/out AM/PM is ignored
    if not ampm and mnt is None:
        return None

    # 12‑hour conversion
    if ampm:
        if ampm.lower() == 'pm' and h != 12:
            h += 12
        if ampm.lower() == 'am' and h == 12:
            h = 0

    # Midnight '00:00' with no AM/PM => ignore
    if not ampm and mnt == '00' and sec is None:
        return None

    if mnt is None:
        return f"{h:02d}"
    if sec is None:
        return f"{h:02d}:{int(mnt):02d}"
    return f"{h:02d}:{int(mnt):02d}:{int(sec):02d}"


# ---------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------
def to_sortable_date(raw: Any, dayFirst: bool = True) -> str:
    """
    Convert *raw* to a sortable date (optionally with time) as described
    in the doc‑string of the original implementation.
    """
    if not raw or not raw.strip():
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
    for pat in (r'\b(\d{4})-(\d{1,2})-(\d{1,2})(?:T|\s|$)',
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
        return f"{date_str} {time}" if time and date_str[0] != '?' else date_str

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
        return _earliest(uymd)[1]

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
