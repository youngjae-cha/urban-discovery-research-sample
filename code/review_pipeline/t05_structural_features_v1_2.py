#!/usr/bin/env python3
"""Exact T05-A structural extractor from prospective lock SHA 116376eb..."""
import re, unicodedata
T05_DURATION={
"explicit_span":[r"\bfor\s+(?:(?:more\s+than|over|nearly|almost)\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|many|several)\s+(?:months?|years?|decades?)\b"],
"since_anchor":[r"\bsince\s+(?:(?:19|20)\d{2}|childhood|i\s+was\s+(?:a\s+)?(?:child|kid|teenager)|we\s+were\s+(?:children|kids|teenagers))\b"],
"calendar_recurrence":[r"\b(?:(?:every|each)\s+(?:week|month|year|summer|winter|spring|fall|autumn|birthday|anniversary)|weekly|monthly|yearly|annually)\b"],
"frequency":[r"\b(?:regularly|frequently|often|countless\s+times|many\s+times|time\s+and\s+again|again\s+and\s+again)\b"],
"continued_return":[r"\b(?:(?:keep|kept|continue|continued|still)\s+(?:coming|going|visiting|returning)|(?:have|has|had|i've|we've)\s+been\s+(?:coming|going|visiting|returning)|come\s+back|go\s+back|returned|returning)\b"],
"tradition_ritual":[r"\b(?:tradition|traditional\s+stop|ritual|annual\s+(?:trip|visit|stop|outing))\b"]}
T05_RELATIONSHIP={
"parent_grandparent":[r"\b(?:mother|mom|mum|father|dad|parents?|grandmother|grandma|grandfather|grandpa|grandparents?)\b"],"spouse_partner":[r"\b(?:husband|wife|spouse|partner|boyfriend|girlfriend|fiance|fiancee|first\s+date)\b"],"child_descendant":[r"\b(?:sons?|daughters?|children|kids?|grandsons?|granddaughters?|grandchildren)\b"],"sibling":[r"\b(?:brothers?|sisters?|siblings?)\b"],"extended_family":[r"\b(?:aunts?|uncles?|cousins?|nieces?|nephews?)\b"],"family_general":[r"\b(?:family|families|loved\s+ones?)\b"],"friend":[r"\b(?:best\s+friends?|friends?)\b"]}
T05_PAST={
"memory":[r"\b(?:remember|remembered|remembering|memories|memory|nostalgia|nostalgic|reminisce|reminiscing)\b"],"childhood":[r"\b(?:childhood|grew\s+up|growing\s+up|as\s+a\s+(?:child|kid|teenager))\b"],"past_anchor":[r"\b(?:years?\s+ago|decades?\s+ago|back\s+then|when\s+i\s+was|when\s+we\s+were|used\s+to)\b"],"milestone":[r"\b(?:my\s+first|our\s+first|first\s+date|wedding|honeymoon|graduation|birthday|anniversary)\b"],"life_history":[r"\b(?:all\s+my\s+life|my\s+whole\s+life|part\s+of\s+my\s+life|part\s+of\s+our\s+lives|throughout\s+my\s+life)\b"]}
OUTPUT_COLUMNS=["t05_duration_recurrence_category_count","t05_relationship_category_count","t05_autobiographical_past_category_count"]
def normalize(value):
    text=unicodedata.normalize("NFKC","" if value is None else str(value)).replace("’","'").replace("‘","'").replace("\r\n","\n").replace("\r","\n").casefold()
    layout="\n".join(re.sub(r"[\t ]+"," ",line).strip() for line in text.split("\n"));return re.sub(r"\s+"," ",layout).strip()
def count(text,defs): return sum(any(re.search(p,text) is not None for p in patterns) for patterns in defs.values())
def extract_t05(text):
    x=normalize(text);return {OUTPUT_COLUMNS[0]:count(x,T05_DURATION),OUTPUT_COLUMNS[1]:count(x,T05_RELATIONSHIP),OUTPUT_COLUMNS[2]:count(x,T05_PAST)}
