#!/usr/bin/env python3
"""Streaming enrichment and corpus-global exact-text duplicate accounting.

This program starts from the fail-closed Parquet files emitted by
``align_state_year_reviews.py``.  It never embeds text, calls an API, fits a
model, or reads human labels/protected outcomes.

The workflow has five explicit, restartable commands:

1. ``build-source-list`` converts the passing all-state preflight receipt into
   an immutable expected source inventory.
2. ``stage-shard`` streams one aligned state/year and appends the exact frozen
   ``n_words``, ``generic_only_flag``, ``normalized_text_hash``, and a
   deterministic duplicate bucket.
3. ``partition-corpus`` verifies the complete expected shard inventory and
   externally partitions it by duplicate bucket with DuckDB.
4. ``finalize-bucket`` independently computes the corpus-global
   ``normalized_duplicate_count_within_place`` for one bucket.  Equal
   ``(gmap_id, normalized_text_hash)`` keys are guaranteed to share a bucket.
5. ``verify-corpus`` verifies every bucket receipt and publishes a final
   corpus receipt.

No command loads the full corpus in RAM.  Publication is atomic at the shard,
partitioned-corpus, and finalized-bucket levels.  Existing outputs are reused
only when their receipts and hashes match; inconsistent outputs fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_VERSION = "1.0.0"
POLICY_VERSION = "frozen-common-text-policy-v1-exact-20260831"
PARTITION_VERSION = "global-normalized-text-dedup-v1"
ALIGNMENT_STATUS = "complete_state_year_exact_key_alignment"

SCORE_COLUMNS = [
    "Happiness_Score",
    "Meaningfulness_Score",
    "Psychological_Richness_Score",
    "FG_Richness_Score",
    "Exploration_Behavior_Score",
    "Satisfaction_SWLS_Score",
    "Meaningfulness_MLQ_Score",
    "Purpose_MLQ_Score",
    "Appreciation_Score",
    "Mattering_Score",
    "Coherence_Score",
    "PR_Scale_Overall_Score",
    "Transformative_Experience_Score",
    "Curiosity_Stretching_Score",
    "Curiosity_Embracing_Score",
    "Self_Expansion_Score",
    "Intolerance_of_Uncertainty_Score",
    "Learning_Knowledge_Score",
    "Openness_to_Uncertainty_Score",
    "Need_for_Cognition_Score",
    "Vitality_Aliveness_Score",
    "Cosmopolitanism_Score",
    "Shared_Reality_Score",
    "Perspective_Change_Score",
    "Surprise_Score",
    "SPANE_PA_Score",
    "SPANE_NA_Score",
    "Price_Benchmark_Score",
    "Service_Speed_Benchmark_Score",
]

# These constants are copied exactly from
# evidence_pipeline_20260831/prepare_evidence_pipeline_data.py.  Do not alter
# them without defining a new policy version and rerunning development QA.
WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", flags=re.UNICODE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "here", "him", "his", "i",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "she",
    "so", "that", "the", "their", "them", "there", "they", "this", "to",
    "too", "very", "was", "we", "were", "with", "you", "your",
}
TEMPORAL = {
    "today", "yesterday", "tonight", "tomorrow", "ago", "before", "after",
    "during", "while", "when", "first", "last", "recently", "later", "early",
    "morning", "afternoon", "evening", "night", "day", "week", "weekend",
    "month", "year", "annual", "return", "returned", "again",
}
EVENT = {
    "went", "came", "visited", "stopped", "walked", "arrived", "ordered",
    "tried", "got", "had", "ate", "drank", "watched", "attended", "stayed",
    "took", "bought", "received", "asked", "told", "waited", "returned",
    "brought", "served", "met", "spoke", "called", "booked", "played",
    "experienced", "happened", "started", "left",
}
COGNITIVE = {
    "think", "thought", "realize", "realized", "learn", "learned", "learnt",
    "understand", "understood", "know", "knew", "notice", "noticed",
    "discover", "discovered", "expect", "expected", "remember", "remembered",
    "wonder", "wondered", "consider", "considered", "decide", "decided",
    "believe", "believed", "figure", "figured", "recognize", "recognized",
    "compare", "compared", "imagine", "imagined",
}
GENERIC_CONTENT = {
    "amazing", "awesome", "bad", "best", "clean", "decent", "delicious",
    "excellent", "fantastic", "fast", "food", "friendly", "good", "great",
    "helpful", "horrible", "location", "love", "loved", "nice", "okay", "ok",
    "people", "perfect", "place", "polite", "poor", "quick", "recommend",
    "recommended", "restaurant", "rude", "service", "spot", "staff", "terrible",
    "tasty", "wonderful", "worst", "worth", "yummy", "atmosphere", "experience",
}
OLD_SCORER_CLEAN_RE = re.compile(r"[^A-Za-z0-9\s.,!'\"]+")
SAFE_SOURCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class IntegrityError(RuntimeError):
    """Raised when an input/output cannot satisfy the frozen contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def policy_fingerprint() -> str:
    payload = {
        "policy_version": POLICY_VERSION,
        "word_re": WORD_RE.pattern,
        "url_re": URL_RE.pattern,
        "email_re": EMAIL_RE.pattern,
        "stopwords": sorted(STOPWORDS),
        "temporal": sorted(TEMPORAL),
        "event": sorted(EVENT),
        "cognitive": sorted(COGNITIVE),
        "generic_content": sorted(GENERIC_CONTENT),
        "normalization": [
            "NFKC", "lower", "curly quote canonicalization", "URL/email tokens",
            "remove non-word except apostrophe angle brackets hyphen", "collapse whitespace",
        ],
        "generic_rule": (
            "n_words<=15 and content_tokens and generic_ratio>=.80 and no "
            "digit/currency/duration/date-clock and temporal=event=cognitive=0"
        ),
    }
    return canonical_hash(payload)


POLICY_SHA256 = policy_fingerprint()


def normalized_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).lower()
    value = value.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    value = URL_RE.sub(" <url> ", value)
    value = EMAIL_RE.sub(" <email> ", value)
    value = re.sub(r"[^\w\s'<>-]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def frozen_text_fields(text: Any) -> tuple[int, bool, str]:
    if text is None or text is pd.NA or (isinstance(text, float) and math.isnan(text)):
        raise IntegrityError("aligned review_text is missing")
    raw = str(text)
    tokens = [token.lower().replace("’", "'") for token in WORD_RE.findall(raw)]
    n_words = len(tokens)
    content_tokens = [token for token in tokens if token not in STOPWORDS]
    generic_ratio = (
        sum(token in GENERIC_CONTENT for token in content_tokens) / len(content_tokens)
        if content_tokens else 0.0
    )
    n_digits = sum(ch.isdigit() for ch in raw)
    has_currency = bool(
        re.search(r"[$€£¥₹]|\b(?:usd|dollars?|bucks?|euros?|pounds?)\b", raw, re.I)
    )
    has_duration = bool(
        re.search(
            r"\b\d+(?:\.\d+)?\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?)\b",
            raw,
            re.I,
        )
    )
    has_date_or_clock = bool(
        re.search(
            r"\b(?:\d{1,2}[:.]\d{2}\s*(?:am|pm)?|\d{1,2}\s*(?:am|pm)|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2})\b",
            raw,
            re.I,
        )
    )
    temporal_count = sum(token in TEMPORAL for token in tokens)
    event_count = sum(token in EVENT for token in tokens)
    cognitive_count = sum(token in COGNITIVE for token in tokens)
    generic_only = bool(
        n_words <= 15
        and content_tokens
        and generic_ratio >= 0.80
        and not (n_digits or has_currency or has_duration or has_date_or_clock)
        and temporal_count == 0
        and event_count == 0
        and cognitive_count == 0
    )
    normalized_hash = sha256_bytes(normalized_text(raw))
    return n_words, generic_only, normalized_hash


def duplicate_bucket(normalized_hash: str, bucket_count: int) -> int:
    # The normalized hash is already a uniformly distributed SHA-256 digest.
    # Equal within-place duplicate keys therefore always share a bucket.
    return int(normalized_hash[:16], 16) % bucket_count


def expected_review_uid(
    state: str,
    source_name: str,
    source_row: int,
    gmap_id: str,
    time_ms: int,
    text_sha256: str,
) -> str:
    payload = json.dumps(
        [state, source_name, source_row, gmap_id, time_ms, text_sha256],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256_bytes(payload)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"cannot read JSON receipt {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntegrityError(f"JSON receipt must be an object: {path}")
    return payload


def validate_source_id(value: str) -> str:
    if not SAFE_SOURCE_ID_RE.fullmatch(value):
        raise ValueError(
            "source id must match [A-Za-z0-9][A-Za-z0-9._-]*; " f"got {value!r}"
        )
    return value


def score_fingerprint(chunks: Iterator[pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for frame in chunks:
        values = frame[SCORE_COLUMNS].to_numpy(dtype="<f4", copy=True)
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def parquet_batches(path: Path, columns: list[str], batch_rows: int) -> Iterator[pd.DataFrame]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=columns):
        yield batch.to_pandas()


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_path_list(paths: list[Path]) -> str:
    if not paths:
        raise IntegrityError("empty Parquet path inventory")
    return "[" + ",".join(sql_string(path.resolve()) for path in paths) + "]"


def configure_duckdb(
    connection: duckdb.DuckDBPyConnection,
    *,
    temp_dir: Path,
    memory_limit: str,
    threads: int,
) -> None:
    if threads <= 0:
        raise ValueError("--threads must be positive")
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory={sql_string(temp_dir.resolve())}")
    connection.execute(f"SET memory_limit={sql_string(memory_limit)}")
    connection.execute(f"SET threads={threads}")
    connection.execute("SET preserve_insertion_order=false")


def load_source_list(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        values = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    else:
        if isinstance(payload, dict):
            payload = payload.get("source_ids")
        if not isinstance(payload, list):
            raise IntegrityError("source list JSON must be an array or {'source_ids': [...]}")
        values = [str(value) for value in payload]
    for value in values:
        validate_source_id(value)
    if not values or len(values) != len(set(values)):
        raise IntegrityError("source list must be nonempty and unique")
    return values


def build_source_list(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite source inventory: {args.output}")
    preflight = read_json(args.preflight_receipt)
    if preflight.get("status") != "PASS_READY_FOR_STATE_YEAR_ALIGNMENT":
        raise IntegrityError("preflight receipt is not passing")
    if preflight.get("all_state_mode") is not True:
        raise IntegrityError("production source inventory requires an all-state preflight")
    year = int(preflight.get("year", -1))
    states = preflight.get("states")
    if (
        not isinstance(states, list)
        or len(states) != int(preflight.get("state_count", -1))
        or len(states) != len(set(map(str, states)))
        or not states
    ):
        raise IntegrityError("preflight state inventory is missing, duplicated, or inconsistent")
    source_ids = [validate_source_id(f"{state}_{year}") for state in states]
    payload = {
        "status": "FROZEN_EXPECTED_SOURCE_INVENTORY",
        "created_utc": utc_now(),
        "preflight_receipt_path": str(args.preflight_receipt.resolve()),
        "preflight_receipt_sha256": sha256_file(args.preflight_receipt),
        "preflight_state_inventory_sha256": preflight.get("state_inventory_sha256"),
        "year": year,
        "source_count": len(source_ids),
        "source_ids": source_ids,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


def output_file_inventory(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*.parquet")):
        rows.append(
            {
                "relative_path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def validate_alignment(input_path: Path, manifest_path: Path) -> tuple[dict[str, Any], str, str]:
    if not input_path.is_file() or input_path.suffix.lower() != ".parquet":
        raise IntegrityError(f"aligned input is not an existing Parquet: {input_path}")
    manifest = read_json(manifest_path)
    if manifest.get("status") != ALIGNMENT_STATUS:
        raise IntegrityError(f"alignment manifest status is not {ALIGNMENT_STATUS}")
    input_hash = sha256_file(input_path)
    if manifest.get("output", {}).get("sha256") != input_hash:
        raise IntegrityError("aligned input SHA-256 does not match alignment manifest")
    names = pq.ParquetFile(input_path).schema.names
    required = {
        "review_uid", "state", "source_row_0based", "gmap_id", "time_ms", "year",
        "review_text", "review_text_sha256", "scorer_clean_text_sha256", *SCORE_COLUMNS,
    }
    missing = sorted(required - set(names))
    if missing:
        raise IntegrityError(f"aligned input lacks required columns: {missing}")
    conflicts = {
        "source_id", "n_words", "generic_only_flag", "normalized_text_hash",
        "dedup_bucket", "normalized_duplicate_count_within_place",
    } & set(names)
    if conflicts:
        raise IntegrityError(f"aligned input already contains enrichment fields: {sorted(conflicts)}")
    expected_rows = int(manifest.get("counts", {}).get("aligned_rows", -1))
    actual_rows = pq.ParquetFile(input_path).metadata.num_rows
    if expected_rows <= 0 or expected_rows != actual_rows:
        raise IntegrityError(
            f"alignment row count mismatch: manifest={expected_rows}, parquet={actual_rows}"
        )
    return manifest, input_hash, sha256_file(manifest_path)


def validate_reusable_stage(
    output: Path,
    receipt_path: Path,
    *,
    input_hash: str,
    manifest_hash: str,
    source_id: str,
    bucket_count: int,
) -> dict[str, Any] | None:
    if not output.exists() and not receipt_path.exists():
        return None
    if not output.is_file() or not receipt_path.is_file():
        raise IntegrityError("partial pre-existing stage output; use a clean work directory")
    receipt = read_json(receipt_path)
    expected = {
        "status": "PASS_COMPLETE_STAGE_SHARD",
        "source_id": source_id,
        "input_sha256": input_hash,
        "alignment_manifest_sha256": manifest_hash,
        "policy_sha256": POLICY_SHA256,
        "bucket_count": bucket_count,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise IntegrityError("pre-existing stage receipt does not match requested run")
    if receipt.get("output_sha256") != sha256_file(output):
        raise IntegrityError("pre-existing staged Parquet hash mismatch")
    return receipt


def stage_shard(args: argparse.Namespace) -> None:
    source_id = validate_source_id(args.source_id)
    if args.bucket_count <= 0 or args.bucket_count > 65_536:
        raise ValueError("--bucket-count must be in [1, 65536]")
    if args.batch_rows <= 0:
        raise ValueError("--batch-rows must be positive")
    manifest, input_hash, manifest_hash = validate_alignment(args.input, args.alignment_manifest)
    stage_dir = args.work_dir / "staged"
    receipt_dir = args.work_dir / "receipts" / "stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    output = stage_dir / f"{source_id}.parquet"
    receipt_path = receipt_dir / f"{source_id}.json"
    reusable = validate_reusable_stage(
        output,
        receipt_path,
        input_hash=input_hash,
        manifest_hash=manifest_hash,
        source_id=source_id,
        bucket_count=args.bucket_count,
    )
    if reusable is not None:
        print(json.dumps({**reusable, "invocation_status": "REUSED_COMPLETE_STAGE_SHARD"}, indent=2))
        return

    source_state = str(manifest.get("state", ""))
    source_year = int(manifest.get("year", -1))
    raw_source = Path(str(manifest.get("source", {}).get("raw_review_gz", "")))
    if not source_state or source_year < 0 or not raw_source.name:
        raise IntegrityError("alignment manifest lacks state/year/raw source provenance")

    parquet = pq.ParquetFile(args.input)
    input_schema = parquet.schema_arrow
    output_schema = pa.schema(
        [
            *list(input_schema),
            pa.field("source_id", pa.string(), nullable=False),
            pa.field("n_words", pa.int32(), nullable=False),
            pa.field("generic_only_flag", pa.bool_(), nullable=False),
            pa.field("normalized_text_hash", pa.string(), nullable=False),
            pa.field("dedup_bucket", pa.int32(), nullable=False),
        ],
        metadata={
            **(input_schema.metadata or {}),
            b"enrichment_policy": POLICY_VERSION.encode(),
            b"enrichment_policy_sha256": POLICY_SHA256.encode(),
            b"dedup_partition_version": PARTITION_VERSION.encode(),
            b"dedup_bucket_count": str(args.bucket_count).encode(),
        },
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".partial", dir=stage_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    writer: pq.ParquetWriter | None = None
    rows = generic_rows = short_rows = 0
    previous_source_row: int | None = None
    identity_digest = hashlib.sha256()
    input_score_digest = hashlib.sha256()
    started = time.perf_counter()
    try:
        writer = pq.ParquetWriter(
            temporary,
            output_schema,
            compression=args.compression,
            use_dictionary=True,
            write_statistics=True,
        )
        for batch_index, batch in enumerate(parquet.iter_batches(batch_size=args.batch_rows)):
            frame = batch.to_pandas()
            if frame.empty:
                continue
            if frame[["review_uid", "gmap_id", "review_text"]].isna().any().any():
                raise IntegrityError(f"batch {batch_index}: missing identity or review text")
            source_rows = pd.to_numeric(frame["source_row_0based"], errors="coerce")
            if source_rows.isna().any() or (source_rows < 0).any() or (source_rows % 1 != 0).any():
                raise IntegrityError(f"batch {batch_index}: invalid source_row_0based")
            source_values = source_rows.to_numpy(dtype=np.int64)
            if np.any(np.diff(source_values) <= 0) or (
                previous_source_row is not None and source_values[0] <= previous_source_row
            ):
                raise IntegrityError("source_row_0based must be globally strictly increasing in shard")
            previous_source_row = int(source_values[-1])
            if not frame["state"].astype(str).eq(source_state).all():
                raise IntegrityError(f"batch {batch_index}: state differs from alignment manifest")
            years = pd.to_numeric(frame["year"], errors="coerce")
            if years.isna().any() or not years.eq(source_year).all():
                raise IntegrityError(f"batch {batch_index}: year differs from alignment manifest")

            score_values = frame[SCORE_COLUMNS].to_numpy(dtype=np.float32, copy=True)
            if not np.isfinite(score_values).all():
                raise IntegrityError(f"batch {batch_index}: non-finite raw cosine score")
            input_score_digest.update(np.asarray(score_values, dtype="<f4").tobytes(order="C"))

            features = frame["review_text"].map(frozen_text_fields)
            frame["n_words"] = features.map(lambda value: value[0]).astype(np.int32)
            frame["generic_only_flag"] = features.map(lambda value: value[1]).astype(bool)
            frame["normalized_text_hash"] = features.map(lambda value: value[2]).astype(str)
            frame["dedup_bucket"] = frame["normalized_text_hash"].map(
                lambda value: duplicate_bucket(value, args.bucket_count)
            ).astype(np.int32)
            frame["source_id"] = source_id

            raw_hash = frame["review_text"].astype(str).map(sha256_bytes)
            if not raw_hash.eq(frame["review_text_sha256"].astype(str)).all():
                first = int(np.flatnonzero(~raw_hash.eq(frame["review_text_sha256"].astype(str)).to_numpy())[0])
                raise IntegrityError(f"batch {batch_index}: review_text_sha256 mismatch at row {first}")
            old_clean_hash = frame["review_text"].astype(str).map(
                lambda text: sha256_bytes(OLD_SCORER_CLEAN_RE.sub("", text))
            )
            if not old_clean_hash.eq(frame["scorer_clean_text_sha256"].astype(str)).all():
                raise IntegrityError(f"batch {batch_index}: scorer_clean_text_sha256 mismatch")

            expected_uids = [
                expected_review_uid(
                    source_state,
                    raw_source.name,
                    int(row),
                    str(place),
                    int(timestamp),
                    str(text_hash),
                )
                for row, place, timestamp, text_hash in zip(
                    source_values,
                    frame["gmap_id"],
                    frame["time_ms"],
                    raw_hash,
                    strict=True,
                )
            ]
            observed_uids = frame["review_uid"].astype(str).tolist()
            if expected_uids != observed_uids:
                first = next(i for i, (a, b) in enumerate(zip(expected_uids, observed_uids)) if a != b)
                raise IntegrityError(f"batch {batch_index}: stable review_uid mismatch at row {first}")
            for uid in observed_uids:
                identity_digest.update(uid.encode("ascii"))
                identity_digest.update(b"\n")

            table = pa.Table.from_pandas(frame, schema=output_schema, preserve_index=False, safe=True)
            writer.write_table(table, row_group_size=args.batch_rows)
            rows += len(frame)
            generic_rows += int(frame["generic_only_flag"].sum())
            short_rows += int(frame["n_words"].le(4).sum())
        writer.close()
        writer = None
        expected_rows = int(manifest["counts"]["aligned_rows"])
        if rows != expected_rows or rows <= 0:
            raise IntegrityError(f"staged row mismatch: staged={rows}, aligned={expected_rows}")
        os.replace(temporary, output)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise

    output_score_digest = score_fingerprint(
        parquet_batches(output, SCORE_COLUMNS, args.batch_rows)
    )
    if input_score_digest.hexdigest() != output_score_digest:
        output.unlink(missing_ok=True)
        raise IntegrityError("29-score binary fingerprint changed during staging")
    elapsed = time.perf_counter() - started
    receipt = {
        "status": "PASS_COMPLETE_STAGE_SHARD",
        "created_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "policy_version": POLICY_VERSION,
        "policy_sha256": POLICY_SHA256,
        "partition_version": PARTITION_VERSION,
        "source_id": source_id,
        "state": source_state,
        "year": source_year,
        "raw_source_name": raw_source.name,
        "input_path": str(args.input.resolve()),
        "input_sha256": input_hash,
        "alignment_manifest_path": str(args.alignment_manifest.resolve()),
        "alignment_manifest_sha256": manifest_hash,
        "alignment_status": manifest["status"],
        "bucket_count": args.bucket_count,
        "rows": rows,
        "generic_only_rows": generic_rows,
        "short_le4_rows": short_rows,
        "identity_sequence_sha256": identity_digest.hexdigest(),
        "raw_29_score_sequence_sha256": output_score_digest,
        "raw_score_count": len(SCORE_COLUMNS),
        "raw_scores_preserved_exactly": True,
        "output_path": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "output_size_bytes": output.stat().st_size,
        "elapsed_seconds": elapsed,
        "no_embedding_api_model_or_labels": True,
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))


def validated_stage_inventory(work_dir: Path, source_list: Path) -> tuple[list[Path], list[dict[str, Any]], str]:
    sources = load_source_list(source_list)
    receipt_dir = work_dir / "receipts" / "stage"
    observed_receipts = {path.stem for path in receipt_dir.glob("*.json")}
    if observed_receipts != set(sources):
        raise IntegrityError(
            "stage receipt inventory differs from source list; "
            f"missing={sorted(set(sources)-observed_receipts)}, "
            f"extra={sorted(observed_receipts-set(sources))}"
        )
    paths: list[Path] = []
    receipts: list[dict[str, Any]] = []
    scopes: set[tuple[str, int]] = set()
    bucket_count: int | None = None
    for source in sources:
        receipt_path = receipt_dir / f"{source}.json"
        receipt = read_json(receipt_path)
        if receipt.get("status") != "PASS_COMPLETE_STAGE_SHARD":
            raise IntegrityError(f"stage {source} is not complete")
        if receipt.get("source_id") != source or receipt.get("policy_sha256") != POLICY_SHA256:
            raise IntegrityError(f"stage {source} provenance/policy mismatch")
        current_buckets = int(receipt.get("bucket_count", -1))
        if bucket_count is None:
            bucket_count = current_buckets
        elif current_buckets != bucket_count:
            raise IntegrityError("staged shards use different bucket counts")
        scope = (str(receipt.get("state")), int(receipt.get("year", -1)))
        if scope in scopes:
            raise IntegrityError(f"duplicate state/year source scope: {scope}")
        scopes.add(scope)
        output = Path(str(receipt.get("output_path", "")))
        if output != (work_dir / "staged" / f"{source}.parquet").resolve():
            raise IntegrityError(f"stage {source} output path differs from work-dir contract")
        if not output.is_file() or sha256_file(output) != receipt.get("output_sha256"):
            raise IntegrityError(f"stage {source} output missing or hash mismatch")
        if pq.ParquetFile(output).metadata.num_rows != int(receipt.get("rows", -1)):
            raise IntegrityError(f"stage {source} Parquet row count mismatch")
        paths.append(output)
        receipts.append(receipt)
    inventory_hash = canonical_hash(
        [
            {
                "source_id": receipt["source_id"],
                "state": receipt["state"],
                "year": receipt["year"],
                "rows": receipt["rows"],
                "output_sha256": receipt["output_sha256"],
            }
            for receipt in receipts
        ]
    )
    return paths, receipts, inventory_hash


def validate_partition_reuse(
    partition_dir: Path,
    inventory_hash: str,
    expected_rows: int | None = None,
) -> dict[str, Any] | None:
    receipt_path = partition_dir / "_PARTITION_RECEIPT.json"
    if not partition_dir.exists():
        return None
    if not partition_dir.is_dir() or not receipt_path.is_file():
        raise IntegrityError("partial pre-existing partition corpus; use a clean work directory")
    receipt = read_json(receipt_path)
    if (
        receipt.get("status") != "PASS_COMPLETE_GLOBAL_PARTITION"
        or receipt.get("stage_inventory_sha256") != inventory_hash
        or receipt.get("policy_sha256") != POLICY_SHA256
    ):
        raise IntegrityError("pre-existing partition receipt does not match staged corpus")
    if expected_rows is not None and (
        receipt.get("expected_rows_contract") != expected_rows
        or int(receipt.get("rows", -1)) != expected_rows
    ):
        raise IntegrityError("pre-existing partition does not match --expected-rows")
    current = output_file_inventory(partition_dir)
    if current != receipt.get("parquet_files"):
        raise IntegrityError("pre-existing partitioned Parquet inventory/hash mismatch")
    return receipt


def partition_corpus(args: argparse.Namespace) -> None:
    paths, receipts, inventory_hash = validated_stage_inventory(args.work_dir, args.source_list)
    if args.expected_rows is not None and args.expected_rows <= 0:
        raise ValueError("--expected-rows must be positive")
    staged_rows = sum(int(receipt["rows"]) for receipt in receipts)
    if args.expected_rows is not None and staged_rows != args.expected_rows:
        raise IntegrityError(
            f"staged corpus has {staged_rows} rows, expected {args.expected_rows}"
        )
    partition_dir = args.work_dir / "partitioned"
    reusable = validate_partition_reuse(
        partition_dir, inventory_hash, expected_rows=args.expected_rows
    )
    if reusable is not None:
        print(json.dumps({**reusable, "invocation_status": "REUSED_COMPLETE_GLOBAL_PARTITION"}, indent=2))
        return
    if partition_dir.exists():
        raise IntegrityError("refusing to overwrite pre-existing partition directory")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".partitioned.", dir=args.work_dir))
    started = time.perf_counter()
    try:
        connection = duckdb.connect()
        configure_duckdb(
            connection,
            temp_dir=args.temp_dir,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        path_sql = sql_path_list(paths)
        connection.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet({path_sql}, union_by_name=false)
            ) TO {sql_string(temporary.resolve())}
            (FORMAT PARQUET, PARTITION_BY (dedup_bucket), COMPRESSION ZSTD,
             ROW_GROUP_SIZE {int(args.row_group_size)})
            """
        )
        glob = str((temporary / "**" / "*.parquet").resolve())
        stats = connection.execute(
            f"""
            SELECT count(*)::BIGINT, min(dedup_bucket)::INTEGER,
                   max(dedup_bucket)::INTEGER, count(DISTINCT dedup_bucket)::INTEGER,
                   count(DISTINCT source_id)::INTEGER
            FROM read_parquet({sql_string(glob)}, hive_partitioning=true)
            """
        ).fetchone()
        buckets = [
            int(row[0])
            for row in connection.execute(
                f"SELECT DISTINCT dedup_bucket FROM read_parquet({sql_string(glob)}, hive_partitioning=true) ORDER BY 1"
            ).fetchall()
        ]
        connection.close()
        expected_rows = staged_rows
        if int(stats[0]) != expected_rows:
            raise IntegrityError(f"partition row mismatch: {stats[0]} vs {expected_rows}")
        bucket_count = int(receipts[0]["bucket_count"])
        if not buckets or min(buckets) < 0 or max(buckets) >= bucket_count:
            raise IntegrityError("partition produced invalid duplicate bucket values")
        if int(stats[4]) != len(receipts):
            raise IntegrityError("partition lost or invented source_id values")
        file_inventory = output_file_inventory(temporary)
        if not file_inventory:
            raise IntegrityError("partition produced no Parquet files")
        receipt = {
            "status": "PASS_COMPLETE_GLOBAL_PARTITION",
            "created_utc": utc_now(),
            "script_version": SCRIPT_VERSION,
            "policy_version": POLICY_VERSION,
            "policy_sha256": POLICY_SHA256,
            "partition_version": PARTITION_VERSION,
            "stage_inventory_sha256": inventory_hash,
            "source_list_path": str(args.source_list.resolve()),
            "source_list_sha256": sha256_file(args.source_list),
            "source_ids": [receipt["source_id"] for receipt in receipts],
            "source_rows": {receipt["source_id"]: int(receipt["rows"]) for receipt in receipts},
            "source_count": len(receipts),
            "rows": expected_rows,
            "expected_rows_contract": args.expected_rows,
            "bucket_count_configured": bucket_count,
            "observed_buckets": buckets,
            "observed_bucket_count": len(buckets),
            "parquet_files": file_inventory,
            "memory_limit": args.memory_limit,
            "threads": args.threads,
            "elapsed_seconds": time.perf_counter() - started,
            "no_embedding_api_model_or_labels": True,
        }
        atomic_json(temporary / "_PARTITION_RECEIPT.json", receipt)
        os.replace(temporary, partition_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(receipt, indent=2))


def bucket_input_inventory(
    partition_dir: Path, partition_receipt: dict[str, Any], bucket: int
) -> tuple[list[Path], str]:
    prefix = f"dedup_bucket={bucket}/"
    expected = [row for row in partition_receipt["parquet_files"] if row["relative_path"].startswith(prefix)]
    if not expected:
        raise IntegrityError(f"bucket {bucket} is not an observed bucket")
    paths: list[Path] = []
    for row in expected:
        path = partition_dir / row["relative_path"]
        if not path.is_file() or path.stat().st_size != int(row["size_bytes"]):
            raise IntegrityError(f"bucket {bucket} input missing/size mismatch: {path}")
        if sha256_file(path) != row["sha256"]:
            raise IntegrityError(f"bucket {bucket} input hash mismatch: {path}")
        paths.append(path)
    return paths, canonical_hash(expected)


def validate_bucket_reuse(
    bucket_dir: Path,
    *,
    bucket: int,
    partition_receipt_hash: str,
    input_inventory_hash: str,
) -> dict[str, Any] | None:
    if not bucket_dir.exists():
        return None
    output = bucket_dir / "data.parquet"
    receipt_path = bucket_dir / "receipt.json"
    if not bucket_dir.is_dir() or not output.is_file() or not receipt_path.is_file():
        raise IntegrityError(f"partial pre-existing finalized bucket {bucket}")
    receipt = read_json(receipt_path)
    expected = {
        "status": "PASS_COMPLETE_GLOBAL_DUPLICATE_BUCKET",
        "bucket": bucket,
        "partition_receipt_sha256": partition_receipt_hash,
        "input_inventory_sha256": input_inventory_hash,
        "policy_sha256": POLICY_SHA256,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise IntegrityError(f"pre-existing finalized bucket {bucket} receipt mismatch")
    if receipt.get("output_sha256") != sha256_file(output):
        raise IntegrityError(f"pre-existing finalized bucket {bucket} output hash mismatch")
    return receipt


def finalize_bucket(args: argparse.Namespace) -> None:
    partition_dir = args.work_dir / "partitioned"
    partition_receipt_path = partition_dir / "_PARTITION_RECEIPT.json"
    partition_receipt = read_json(partition_receipt_path)
    if partition_receipt.get("status") != "PASS_COMPLETE_GLOBAL_PARTITION":
        raise IntegrityError("global partition is not complete")
    if args.bucket not in [int(value) for value in partition_receipt.get("observed_buckets", [])]:
        raise IntegrityError(f"bucket {args.bucket} is outside observed bucket inventory")
    partition_receipt_hash = sha256_file(partition_receipt_path)
    paths, input_inventory_hash = bucket_input_inventory(
        partition_dir, partition_receipt, args.bucket
    )
    final_root = args.work_dir / "finalized"
    final_root.mkdir(parents=True, exist_ok=True)
    bucket_dir = final_root / f"bucket-{args.bucket:05d}"
    reusable = validate_bucket_reuse(
        bucket_dir,
        bucket=args.bucket,
        partition_receipt_hash=partition_receipt_hash,
        input_inventory_hash=input_inventory_hash,
    )
    if reusable is not None:
        print(
            json.dumps(
                {**reusable, "invocation_status": "REUSED_COMPLETE_GLOBAL_DUPLICATE_BUCKET"},
                indent=2,
            )
        )
        return
    temporary = Path(tempfile.mkdtemp(prefix=f".bucket-{args.bucket:05d}.", dir=final_root))
    output = temporary / "data.parquet"
    started = time.perf_counter()
    try:
        connection = duckdb.connect()
        configure_duckdb(
            connection,
            temp_dir=args.temp_dir,
            memory_limit=args.memory_limit,
            threads=args.threads,
        )
        path_sql = sql_path_list(paths)
        connection.execute(
            f"""
            COPY (
                SELECT p.*,
                       count(*) OVER (
                           PARTITION BY gmap_id, normalized_text_hash
                       )::BIGINT AS normalized_duplicate_count_within_place
                FROM read_parquet({path_sql}, hive_partitioning=true) AS p
            ) TO {sql_string(output.resolve())}
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {int(args.row_group_size)})
            """
        )
        output_sql = sql_string(output.resolve())
        rows, unique_ids, min_count, max_count = connection.execute(
            f"""
            SELECT count(*)::BIGINT, count(DISTINCT review_uid)::BIGINT,
                   min(normalized_duplicate_count_within_place)::BIGINT,
                   max(normalized_duplicate_count_within_place)::BIGINT
            FROM read_parquet({output_sql})
            """
        ).fetchone()
        bad_clusters = int(
            connection.execute(
                f"""
                SELECT count(*) FROM (
                    SELECT gmap_id, normalized_text_hash, count(*) AS actual,
                           min(normalized_duplicate_count_within_place) AS recorded_min,
                           max(normalized_duplicate_count_within_place) AS recorded_max
                    FROM read_parquet({output_sql})
                    GROUP BY gmap_id, normalized_text_hash
                    HAVING actual <> recorded_min OR actual <> recorded_max
                )
                """
            ).fetchone()[0]
        )
        cluster_count = int(
            connection.execute(
                f"""
                SELECT count(*) FROM (
                    SELECT 1 FROM read_parquet({output_sql})
                    GROUP BY gmap_id, normalized_text_hash
                )
                """
            ).fetchone()[0]
        )
        inverse_sum = float(
            connection.execute(
                f"SELECT sum(1.0 / normalized_duplicate_count_within_place) FROM read_parquet({output_sql})"
            ).fetchone()[0]
        )
        source_rows = {
            str(source): int(count)
            for source, count in connection.execute(
                f"SELECT source_id, count(*)::BIGINT FROM read_parquet({output_sql}) GROUP BY source_id ORDER BY source_id"
            ).fetchall()
        }
        connection.close()
        if rows <= 0 or unique_ids != rows:
            raise IntegrityError(f"bucket {args.bucket}: review_uid is not unique")
        if min_count < 1 or bad_clusters:
            raise IntegrityError(f"bucket {args.bucket}: duplicate-count invariant failed")
        if not math.isclose(inverse_sum, cluster_count, rel_tol=1e-10, abs_tol=1e-7):
            raise IntegrityError(f"bucket {args.bucket}: inverse-count weight identity failed")
        receipt = {
            "status": "PASS_COMPLETE_GLOBAL_DUPLICATE_BUCKET",
            "created_utc": utc_now(),
            "script_version": SCRIPT_VERSION,
            "policy_version": POLICY_VERSION,
            "policy_sha256": POLICY_SHA256,
            "partition_version": PARTITION_VERSION,
            "bucket": args.bucket,
            "partition_receipt_sha256": partition_receipt_hash,
            "input_inventory_sha256": input_inventory_hash,
            "input_file_count": len(paths),
            "rows": int(rows),
            "unique_review_uids": int(unique_ids),
            "unique_within_place_normalized_text_clusters": cluster_count,
            "minimum_duplicate_count": int(min_count),
            "maximum_duplicate_count": int(max_count),
            "sum_inverse_duplicate_count": inverse_sum,
            "source_rows": source_rows,
            "raw_score_count": len(SCORE_COLUMNS),
            "output_sha256": sha256_file(output),
            "output_size_bytes": output.stat().st_size,
            "elapsed_seconds": time.perf_counter() - started,
            "no_embedding_api_model_or_labels": True,
        }
        atomic_json(temporary / "receipt.json", receipt)
        os.replace(temporary, bucket_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(receipt, indent=2))


def verify_corpus(args: argparse.Namespace) -> None:
    _, stage_receipts, inventory_hash = validated_stage_inventory(args.work_dir, args.source_list)
    partition_dir = args.work_dir / "partitioned"
    partition_receipt_path = partition_dir / "_PARTITION_RECEIPT.json"
    partition_receipt = validate_partition_reuse(partition_dir, inventory_hash)
    if partition_receipt is None:
        raise IntegrityError("global partition is absent")
    partition_receipt_hash = sha256_file(partition_receipt_path)
    observed_buckets = [int(value) for value in partition_receipt["observed_buckets"]]
    final_root = args.work_dir / "finalized"
    expected_dirs = {f"bucket-{bucket:05d}" for bucket in observed_buckets}
    observed_dirs = {path.name for path in final_root.glob("bucket-*") if path.is_dir()}
    if observed_dirs != expected_dirs:
        raise IntegrityError(
            "finalized bucket inventory mismatch; "
            f"missing={sorted(expected_dirs-observed_dirs)}, extra={sorted(observed_dirs-expected_dirs)}"
        )
    total_rows = total_clusters = 0
    source_rows: dict[str, int] = {receipt["source_id"]: 0 for receipt in stage_receipts}
    bucket_outputs = []
    for bucket in observed_buckets:
        bucket_dir = final_root / f"bucket-{bucket:05d}"
        output = bucket_dir / "data.parquet"
        receipt = read_json(bucket_dir / "receipt.json")
        paths, input_inventory_hash = bucket_input_inventory(
            partition_dir, partition_receipt, bucket
        )
        expected = {
            "status": "PASS_COMPLETE_GLOBAL_DUPLICATE_BUCKET",
            "bucket": bucket,
            "partition_receipt_sha256": partition_receipt_hash,
            "input_inventory_sha256": input_inventory_hash,
            "policy_sha256": POLICY_SHA256,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise IntegrityError(f"finalized bucket {bucket} receipt mismatch")
        if not output.is_file() or sha256_file(output) != receipt.get("output_sha256"):
            raise IntegrityError(f"finalized bucket {bucket} output hash mismatch")
        if pq.ParquetFile(output).metadata.num_rows != int(receipt["rows"]):
            raise IntegrityError(f"finalized bucket {bucket} row count mismatch")
        total_rows += int(receipt["rows"])
        total_clusters += int(receipt["unique_within_place_normalized_text_clusters"])
        for source, count in receipt["source_rows"].items():
            if source not in source_rows:
                raise IntegrityError(f"bucket {bucket} contains unknown source {source}")
            source_rows[source] += int(count)
        bucket_outputs.append(
            {
                "bucket": bucket,
                "path": str(output.resolve()),
                "sha256": receipt["output_sha256"],
                "rows": int(receipt["rows"]),
            }
        )
    expected_source_rows = {
        receipt["source_id"]: int(receipt["rows"]) for receipt in stage_receipts
    }
    if source_rows != expected_source_rows:
        raise IntegrityError(
            f"final source row counts differ from stage: {source_rows} vs {expected_source_rows}"
        )
    if total_rows != int(partition_receipt["rows"]):
        raise IntegrityError("final corpus total rows differ from partitioned corpus")
    receipt = {
        "status": "PASS_READY_FOR_FROZEN_EVIDENCE_SCORING_ENGINEERING_ONLY",
        "created_utc": utc_now(),
        "script_version": SCRIPT_VERSION,
        "policy_version": POLICY_VERSION,
        "policy_sha256": POLICY_SHA256,
        "partition_version": PARTITION_VERSION,
        "stage_inventory_sha256": inventory_hash,
        "partition_receipt_sha256": partition_receipt_hash,
        "rows": total_rows,
        "source_rows": source_rows,
        "source_count": len(source_rows),
        "observed_bucket_count": len(observed_buckets),
        "unique_within_place_normalized_text_clusters": total_clusters,
        "bucket_outputs": bucket_outputs,
        "raw_score_count": len(SCORE_COLUMNS),
        "global_review_uid_uniqueness_basis": (
            "Each UID is recomputed from state/raw-source/source-row/gmap/time/text hash; "
            "source rows are strictly increasing within a shard and state/year scopes are unique."
        ),
        "global_duplicate_definition": "exact frozen normalized text within gmap_id across every listed shard",
        "duplicate_weight_for_downstream": "1 / normalized_duplicate_count_within_place",
        "no_embedding_api_model_or_labels": True,
        "scientific_status": (
            "Engineering-ready input for the frozen development-selected evidence scorer; "
            "not independent construct validation."
        ),
    }
    atomic_json(args.work_dir / "FINAL_ENRICHED_CORPUS_RECEIPT.json", receipt)
    print(json.dumps(receipt, indent=2))


def add_duckdb_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--memory-limit", default="16GB")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--temp-dir", type=Path, required=True)
    parser.add_argument("--row-group-size", type=int, default=500_000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    source_list = subparsers.add_parser(
        "build-source-list", help="freeze source IDs from a passing all-state preflight"
    )
    source_list.add_argument("--preflight-receipt", type=Path, required=True)
    source_list.add_argument("--output", type=Path, required=True)
    source_list.set_defaults(function=build_source_list)

    stage = subparsers.add_parser("stage-shard", help="stream and enrich one aligned shard")
    stage.add_argument("--input", type=Path, required=True)
    stage.add_argument("--alignment-manifest", type=Path, required=True)
    stage.add_argument("--source-id", required=True)
    stage.add_argument("--work-dir", type=Path, required=True)
    stage.add_argument("--bucket-count", type=int, default=256)
    stage.add_argument("--batch-rows", type=int, default=100_000)
    stage.add_argument("--compression", choices=["zstd", "snappy", "gzip"], default="zstd")
    stage.set_defaults(function=stage_shard)

    partition = subparsers.add_parser(
        "partition-corpus", help="externally partition the exact staged corpus"
    )
    partition.add_argument("--work-dir", type=Path, required=True)
    partition.add_argument("--source-list", type=Path, required=True)
    partition.add_argument(
        "--expected-rows",
        type=int,
        help="Optional frozen corpus row-count contract; use 106030035 for recorded 2019.",
    )
    add_duckdb_options(partition)
    partition.set_defaults(function=partition_corpus)

    finalize = subparsers.add_parser(
        "finalize-bucket", help="compute global within-place duplicate counts for one bucket"
    )
    finalize.add_argument("--work-dir", type=Path, required=True)
    finalize.add_argument("--bucket", type=int, required=True)
    add_duckdb_options(finalize)
    finalize.set_defaults(function=finalize_bucket)

    verify = subparsers.add_parser(
        "verify-corpus", help="verify every finalized bucket and publish final receipt"
    )
    verify.add_argument("--work-dir", type=Path, required=True)
    verify.add_argument("--source-list", type=Path, required=True)
    verify.set_defaults(function=verify_corpus)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # concise fail-closed CLI contract
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
