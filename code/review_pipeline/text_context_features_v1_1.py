#!/usr/bin/env python3
"""Exact six-feature subset of evidence-pipeline text features (2026-08-31 v1)."""
from __future__ import annotations
import math, re, unicodedata

WORD_RE=re.compile(r"[^\W_]+(?:['’-][^\W_]+)*",flags=re.UNICODE)
URL_RE=re.compile(r"https?://\S+|www\.\S+",flags=re.I)
EMAIL_RE=re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
FIRST_PERSON={"i","i'm","im","i've","ive","i'd","me","my","mine","myself","we","we're","we've","weve","we'd","us","our","ours","ourselves"}
TEMPORAL={"today","yesterday","tonight","tomorrow","ago","before","after","during","while","when","first","last","recently","later","early","morning","afternoon","evening","night","day","week","weekend","month","year","annual","return","returned","again"}
EVENT={"went","came","visited","stopped","walked","arrived","ordered","tried","got","had","ate","drank","watched","attended","stayed","took","bought","received","asked","told","waited","returned","brought","served","met","spoke","called","booked","played","experienced","happened","started","left"}
COGNITIVE={"think","thought","realize","realized","learn","learned","learnt","understand","understood","know","knew","notice","noticed","discover","discovered","expect","expected","remember","remembered","wonder","wondered","consider","considered","decide","decided","believe","believed","figure","figured","recognize","recognized","compare","compared","imagine","imagined"}
AFFECTIVE={"feel","feels","feeling","felt","love","loved","hate","hated","enjoy","enjoyed","happy","sad","excited","disappointed","frustrated","amazed","comfortable","relaxed","worried","angry","thankful","grateful","satisfied","dissatisfied","disgusted","surprised","delighted","upset","pleased","calm","anxious"}
STOPWORDS={"a","an","and","are","as","at","be","been","but","by","for","from","had","has","have","he","her","here","him","his","i","in","is","it","its","me","my","of","on","or","our","she","so","that","the","their","them","there","they","this","to","too","very","was","we","were","with","you","your"}
GENERIC_CONTENT={"amazing","awesome","bad","best","clean","decent","delicious","excellent","fantastic","fast","food","friendly","good","great","helpful","horrible","location","love","loved","nice","okay","ok","people","perfect","place","polite","poor","quick","recommend","recommended","restaurant","rude","service","spot","staff","terrible","tasty","wonderful","worst","worth","yummy","atmosphere","experience"}

OUTPUT_COLUMNS=["ctx_generic_only_flag","ctx_generic_content_ratio","ctx_first_person_flag","ctx_event_marker_flag","ctx_cognitive_marker_flag","ctx_affective_marker_flag"]

def normalized_text(text: object)->str:
    if text is None or (isinstance(text,float) and math.isnan(text)): return ""
    value=unicodedata.normalize("NFKC",str(text)).lower().replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"')
    value=URL_RE.sub(" <url> ",value); value=EMAIL_RE.sub(" <email> ",value)
    value=re.sub(r"[^\w\s'<>-]+"," ",value,flags=re.UNICODE)
    return re.sub(r"\s+"," ",value).strip()

def extract_context(text: object)->dict[str,float]:
    raw="" if text is None or (isinstance(text,float) and math.isnan(text)) else str(text)
    tokens=[t.lower().replace("’","'") for t in WORD_RE.findall(raw)]
    content=[t for t in tokens if t not in STOPWORDS]
    ratio=sum(t in GENERIC_CONTENT for t in content)/len(content) if content else 0.0
    temporal=sum(t in TEMPORAL for t in tokens); event=sum(t in EVENT for t in tokens); cognitive=sum(t in COGNITIVE for t in tokens)
    n_digits=sum(ch.isdigit() for ch in raw)
    has_currency=bool(re.search(r"[$€£¥₹]|\b(?:usd|dollars?|bucks?|euros?|pounds?)\b",raw,re.I))
    has_duration=bool(re.search(r"\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b",raw,re.I))
    has_date=bool(re.search(r"\b(?:\d{1,2}[:.]\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2})\b",raw,re.I))
    generic=bool(len(tokens)<=15 and content and ratio>=.80 and not(n_digits or has_currency or has_duration or has_date) and temporal==0 and event==0 and cognitive==0)
    return {"ctx_generic_only_flag":float(generic),"ctx_generic_content_ratio":float(ratio),"ctx_first_person_flag":float(any(t in FIRST_PERSON for t in tokens)),"ctx_event_marker_flag":float(event>0),"ctx_cognitive_marker_flag":float(cognitive>0),"ctx_affective_marker_flag":float(any(t in AFFECTIVE for t in tokens))}

