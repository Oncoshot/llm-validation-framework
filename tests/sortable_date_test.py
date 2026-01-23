import pytest
from shared.sortable_date import to_sortable_date

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