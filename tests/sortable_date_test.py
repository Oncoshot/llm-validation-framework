import pytest
from llmvalidate.sortable_date import to_sortable_date

@pytest.mark.parametrize("raw,expected,dayFirst", [
    # ---- Unambiguous (dayFirst irrelevant; default True) ----
    ("2022", "2022", True),
    ("  The year was 1999  ", "1999", True),
    ("March 2021", "2021-03", True),
    ("March 2021 13:00", "2021-03", True),           # month present, no day → drop time
    ("2021-3", "2021-03", True),
    ("2021/03", "2021-03", True),
    ("05/2021", "2021-05", True),                    # month/year with slash
    ("2021.12", "2021-12", True),                    # dotted month-year
    ("Jan-2024", "2024-01", True),                   # month-year with dash, month abbreviation ("Jan") instead of numeric
    ("Jan/2024", "2024-01", True),                   # month/year with slash, month abbreviation ("Jan") instead of numeric
    ("deadline: 2024-01", "2024-01", True),

    ("20 Feb 2012", "2012-02-20", True),
    ("Feb 20, 2012", "2012-02-20", True),
    ("5th Jan 2020", "2020-01-05", True),
    ("Thu, 4 Jul 2019", "2019-07-04", True),
    ("2019-07-04", "2019-07-04", True),

    ("2011.04.03 08:09", "2011-04-03 08:09", True),        
    ("2011/4/3 8:9:7", "2011-04-03 08:09:07", True),          
    ("2019-07-04 9pm", "2019-07-04 21", True),          
    ("2019-07-04 09:05", "2019-07-04 09:05", True),        
    ("2019-07-04T09:05:07", "2019-07-04 09:05:07", True),     
    ("20210405", "2021-04-05", True),                # compact yyyymmdd

    ("2019-07-04 21:05Z", "2019-07-04 21:05", True),       
    ("2019-07-04T21:05:07+08:00", "2019-07-04 21:05:07", True),
    ("2020-01-02T03:04:05.678Z", "2020-01-02 03:04:05", True),

    ("September 1, 2020 12:00 AM", "2020-09-01 00:00", True), 
    ("Updated at ~~~ 2022-12-31   23:59:59   ###", "2022-12-31 23:59:59", True), 
    ("Report generated on 2023-11-02 at 07:03:59", "2023-11-02 07:03:59", True), 

    ("2024-02-29", "2024-02-29", True),              # leap day
    ("29 Feb 2024 23:59:59 +0100", "2024-02-29 23:59:59", True), 

    # Missing day/month ⇒ drop time
    ("2021-03 13:00 +0800", "2021-03", True),
    ("2021 13:00", "2021", True),

    # Bare-hour vs explicit-hour 
    ("Oct-31-2021 23", "2021-10-31", True),          # bare hour → drop time 
    ("Jan. 02, 2021 7 AM", "2021-01-02 07", True),      # explicit hour

    # Timezone labels ignored (no shifting)
    ("sept 9 2021 07:08 UTC", "2021-09-09 07:08", True),   
    ("2024-03-01T05:06:07+0000 (UTC)", "2024-03-01 05:06:07", True), 

    # Fuzzy/noisy + first-date extraction
    ("[2021-07-01 00:00]", "2021-07-01", True),      # brackets around date
    ("Invoice 123 dated 2024-10-05 08:09:10 id 456", "2024-10-05 08:09:10", True), 
    ("2020-12 to 2021-02", "2020-12", True),

    # Unknown year handling
    ("April 5", "????-04-05", True),
    ("5 April", "????-04-05", True),
    ("Apr 05", "????-04-05", True),
    ("on the 2nd of May", "????-05-02", True),
    ("Feb 29", "????-02-29", True),                  # leap day without year
    ("May", "????-05", True),
    ("Sept", "????-09", True),
    ("Sep", "????-09", True),

    # Two-digit year
    ("Apr 5, '21", "2021-04-05", True),
    ("05 Apr 21", "2021-04-05", True),

    # Month Year (2 digit year)
    ("Apr-21", "2021-04", True),                    #Month + 2 digit year with dash
    ("Jan/24", "2024-01", True),                    #Month + 2 digit year with slash
    ("Jan '24", "2024-01", True),                   #Month + 2 digit year with apostrophe
    ("Jan'24", "2024-01", True),                    #Month + 2 digit year with apostrophe
    ("Jan' 24", "2024-01", True),                   #Month + 2 digit year with straight apostrophe
    ("Jan’ 24", "2024-01", True),                   #Month + 2 digit year with curly apostrophe

    # ISO week dates
    ("2021-W33-5", "2021-08-20", True),
    ("2021-W01-1", "2021-01-04", True),

    # Quarters collapse to year
    ("Q4 2020", "2020", True),

    # Invalid and no-parse
    ("2021-02-29", None, True),                      # invalid non-leap day
    ("2021-13", None, True),                         # invalid month
    ("2021-04-31", None, True),                      # invalid day
    ("2021-00-05", None, True),                      # invalid month zero
    ("2021-04-00", None, True),                      # invalid day zero
    ("13:00", None, True),
    ("", None, True),
    ("   ", None, True),

    # ---- Ambiguous slash dates (exercise both dayFirst=False/True) ----
    ("04/05/2021 13:14", "2021-04-05 13:14", False),       
    ("04/05/2021 13:14", "2021-05-04 13:14", True),        

    ("01/02/2021 01:02", "2021-01-02 01:02", False),       
    ("01/02/2021 01:02", "2021-02-01 01:02", True),        

    ("07/08/2019 09:10", "2019-07-08 09:10", False),       
    ("07/08/2019 09:10", "2019-08-07 09:10", True),        

    ("09/10/2021 08:07", "2021-09-10 08:07", False),       
    ("09/10/2021 08:07", "2021-10-09 08:07", True),        
])
def test_date_only_parse(raw, expected, dayFirst):
    assert to_sortable_date(raw, dayFirst=dayFirst) == expected


# ONC-12551: an hour must not be read as a 2-digit year, and an unknown-year date may carry
# a time. Before the fix these returned a confidently wrong calendar year.
@pytest.mark.parametrize("raw,expected", [
    ("Feb 20 13:45", "????-02-20 13:45"),      # was '2013-02-20'
    ("Feb 20 11 PM", "????-02-20 23"),         # was '2011-02-20'
    ("20 Feb 14:30", "????-02-20 14:30"),      # was '2014-02-20'
    ("Apr 5 13:45", "????-04-05 13:45"),
    ("Apr 5 13:45:09", "????-04-05 13:45:09"),

    # A genuine 2-digit year is still a year — nothing time-like follows it.
    ("Feb 20, 12", "2012-02-20"),
    ("Apr 5, '21", "2021-04-05"),
    ("5 Jan 20", "2020-01-05"),

    # Unknown-year forms with no time keep their previous shape.
    ("April 5", "????-04-05"),
    ("Feb 29", "????-02-29"),
])
def test_hour_is_not_mistaken_for_a_two_digit_year(raw, expected):
    assert to_sortable_date(raw) == expected


# ONC-12551: `h` / `hr` / `hrs` / `hour(s)` cues mark a lone hour as intentional, as the
# module docstring has always claimed. A bare number still does not.
@pytest.mark.parametrize("raw,expected", [
    ("Oct-31-2021 23 h", "2021-10-31 23"),
    ("Oct-31-2021 23 hr", "2021-10-31 23"),
    ("Oct-31-2021 23 hrs", "2021-10-31 23"),
    ("Oct-31-2021 07 hour", "2021-10-31 07"),
    ("Oct-31-2021 around 07 hours", "2021-10-31 07"),
    ("31 Oct 2021 07 hours", "2021-10-31 07"),

    ("Oct-31-2021 23", "2021-10-31"),          # bare hour: still dropped
    ("Oct-31-2021 11 PM", "2021-10-31 23"),    # AM/PM: unchanged
    ("Oct-31-2021 hospital visit", "2021-10-31"),   # 'h' word is not an hour cue
])
def test_explicit_hour_cues(raw, expected):
    assert to_sortable_date(raw) == expected


# ONC-12551: non-text input yields None rather than AttributeError. float('nan') is truthy,
# so a NaN straight out of a DataFrame column used to reach .strip() and raise.
@pytest.mark.parametrize("raw", [5, 20210405, float("nan"), None, [], {}, object()])
def test_non_string_input_returns_none(raw):
    assert to_sortable_date(raw) is None


# Copilot review of PR #18, findings 2/3/5/6 — all pre-existing. An impossible clock value
# drops the *time* and keeps the date, the same way a bare hour does; it does not fail the
# whole parse (only an invalid explicit *date* does, per rule 7).
@pytest.mark.parametrize("raw,expected", [
    # Time components are range-checked.
    ("2021-01-01 29:75:90", "2021-01-01"),      # was '2021-01-01 29:75:90'
    ("2021-01-01 23:30 PM", "2021-01-01"),      # was '2021-01-01 35:30'
    ("2021-01-01 25:00", "2021-01-01"),
    ("2021-01-01 12:60", "2021-01-01"),
    ("2021-01-01 12:30:61", "2021-01-01"),
    ("2021-01-01 11:30 PM", "2021-01-01 23:30"),    # valid 12-hour input unaffected

    # Only *actual* midnight is dropped; whole-hour stamps keep their time.
    ("2021-01-01 13:00", "2021-01-01 13:00"),   # was '2021-01-01'
    ("2021-01-01 09:00", "2021-01-01 09:00"),
    ("2021-01-01 00:00", "2021-01-01"),
    ("September 1, 2020 12:00 AM", "2020-09-01 00:00"),

    # Prose punctuation after a date must not cost the day.
    ("Date: 2021-07-04.", "2021-07-04"),        # was '2021-07'
    ("Seen on 2021-07-04, discharged", "2021-07-04"),
    ("(2021-07-04)", "2021-07-04"),
    ("2021-02-29,", None),                      # invalid: fails, was '2021-02'
    ("2019-07-04T21:05:07+08:00", "2019-07-04 21:05:07"),   # T form still works

    # AM/PM is A-or-P followed by M, not any two of A/P/M.
    ("2021-01-01 7 PP", "2021-01-01"),          # was '2021-01-01 07'
    ("2021-01-01 7 AA", "2021-01-01"),
    ("2021-01-01 7 MP", "2021-01-01"),
    ("2021-01-01 7 AM", "2021-01-01 07"),
    ("2021-01-01 7 pm", "2021-01-01 19"),
])
def test_clock_validation_and_date_boundaries(raw, expected):
    assert to_sortable_date(raw) == expected
