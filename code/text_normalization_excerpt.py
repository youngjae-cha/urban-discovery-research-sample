"""Unchanged text-normalization function and its two pattern dependencies.
Private feature dictionaries and classification parameters are not included.
"""
import math
import re
import unicodedata

URL_RE=re.compile(r"https?://\S+|www\.\S+",flags=re.I)
EMAIL_RE=re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")


def normalized_text(text: object)->str:
    if text is None or (isinstance(text,float) and math.isnan(text)): return ""
    value=unicodedata.normalize("NFKC",str(text)).lower().replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"')
    value=URL_RE.sub(" <url> ",value); value=EMAIL_RE.sub(" <email> ",value)
    value=re.sub(r"[^\w\s'<>-]+"," ",value,flags=re.UNICODE)
    return re.sub(r"\s+"," ",value).strip()
