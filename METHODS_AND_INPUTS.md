# Method source and input contracts

## What is original

`review_pipeline/` contains complete frozen alignment, enrichment, feature-policy, scorer-chain, annual aggregation and spatial files. The semantic encoder module carries the actual 29 seed definitions, helper bodies, centroid loop and original review-chunk loop. Its new input/output/model arguments replace institutional path setup and automatic notebook execution. Numerical operations, review cleaning, original normalized review embeddings, float16 stored scores and the order of seed versus review encoding are retained.

The small v1.2.x scorer files retain inherited validation contracts; the final v1.2.7 entry point uses the v1.2.2 numerical scoring body. They are retained so the original artifact/program hash checks remain intact. Legacy `2018` spatial output-column names are preserved from the source audit; the supplied input frame determines the analysis year.

The four displayed two-pole banks in the walkthrough are complete excerpts from those definitions, not invented review examples. There are 27 two-pole dimensions and two positive-bank-centroid-only dimensions (`SPANE_PA`, `SPANE_NA`). Prototype sentences define semantic references; they are not human-coded review records or validated questionnaire measurements by themselves.

## Stage contracts

| Stage | Required external material |
|---|---|
| Semantic generation | Authorized Google Review JSON-lines source with `text`, `time` in milliseconds and `gmap_id`; an authorized BGE encoder |
| Alignment | Original review source and ordered score chunks, with expected source/year/row counts supplied through the original CLI |
| Enrichment | All annual aligned source shards and source inventory; full-corpus duplicate and UID checks precede scoring |
| Frozen classifier | 29 semantic scores, original text/context fields, actual duplicate multiplicity, and the external fitted artifact bound by hash |
| Annual aggregation | Complete scored buckets, production manifests, sealed enrichment receipt, model artifact and frozen specification |
| Coverage | Eligible place scores, broad venue type, fixed metropolitan membership and coordinates |

These are schema names, not public data records. The classifier CLI validates the original artifact and program contracts; no weights are substituted. The annual aggregation retains the source's 52-source and receipt checks.

## Interpretation and historical limits

Duplicate counting is by place and exact normalized text across all sources of the defined year, before the word-count and generic-content exclusions. Each row receives inverse multiplicity. Eligible positive weighted mass is divided by eligible weighted mass; a partly eligible cluster need not contribute unit mass. At least 50 contributing eligible clusters define the place cohort. Within-type midpoint percentile at least .90 selects discovery places; coverage uses another discovery place in the same metro within 1 km great-circle distance.

The historical encoder's chunk-count resume logic and later manifest-bound corpus checks are different layers. The source is supplied for inspection rather than as a new cloud launcher. Row-key matching and full linkage must not be inferred merely from a first-chunk check; row order within identical restored keys has a documented identification limit. Frozen place metadata do not supply historical coordinates.

Source hashes support retrospective lineage; they do not attest the encoder binary at the time each historical score was created. The frozen code's provisional/development claim ceilings are retained. No new classifier fitting, label validation, embeddings, place scores, coverage or outcome regressions were performed for this release candidate.
