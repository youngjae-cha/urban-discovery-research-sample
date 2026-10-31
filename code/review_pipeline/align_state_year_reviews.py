#!/usr/bin/env python3
"""Fail-closed reconstruction of raw review text plus historical cosine scores.

This program aligns one Google Local state/year at a time.  It deliberately
replays the historical scorer's pandas JSONL chunk size, UTC-like epoch date
derivation, year filter, ASCII-oriented text cleaner, and nonblank rule.  Each
nonempty replayed raw chunk must match exactly one densely numbered historical
score chunk on row count and on every recoverable key:

    gmap_id, year, month, week

No Parquet output is published unless the complete state/year passes.  A
temporary file is removed on any error.  This is reconstruction only: it does
not use coder labels, protected holdouts, ratings, or place-type information.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - environment preflight catches this
    raise SystemExit("pyarrow is required: install pandas, numpy, and pyarrow") from exc


PROTOCOL_VERSION = "googlelocal-score-alignment-v1"
HISTORICAL_RAW_CHUNKSIZE = 100_000
CLEAN_PATTERN = r"[^A-Za-z0-9\s.,!'\"]+"
CHUNK_RE = re.compile(r"chunk_(\d+)\.csv$")
KEY_COLUMNS = ["gmap_id", "year", "month", "week"]
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

OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("review_uid", pa.string(), nullable=False),
        pa.field("state", pa.string(), nullable=False),
        pa.field("source_row_0based", pa.int64(), nullable=False),
        pa.field("raw_chunk_0based", pa.int32(), nullable=False),
        pa.field("row_within_raw_chunk_0based", pa.int32(), nullable=False),
        pa.field("score_chunk_ordinal", pa.int32(), nullable=False),
        pa.field("gmap_id", pa.string(), nullable=False),
        pa.field("time_ms", pa.int64(), nullable=False),
        pa.field("year", pa.int16(), nullable=False),
        pa.field("month", pa.int8(), nullable=False),
        pa.field("week", pa.int8(), nullable=False),
        pa.field("review_text", pa.string(), nullable=False),
        pa.field("review_text_sha256", pa.string(), nullable=False),
        pa.field("scorer_clean_text_sha256", pa.string(), nullable=False),
        *[pa.field(name, pa.float32(), nullable=False) for name in SCORE_COLUMNS],
    ]
)


class IntegrityError(RuntimeError):
    """Raised when source assets do not support exact reconstruction."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--raw-review-gz", required=True, type=Path)
    parser.add_argument("--score-dir", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Default: OUTPUT_PARQUET.manifest.json",
    )
    parser.add_argument(
        "--compression",
        choices=["zstd", "snappy", "gzip"],
        default="zstd",
    )
    return parser.parse_args()


def sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def review_uid(
    state: str,
    source_name: str,
    source_row: int,
    gmap_id: str,
    time_ms: int,
    text_sha256: str,
) -> str:
    # Canonical JSON avoids delimiter ambiguity.  The identifier is stable only
    # while the original source file and row order remain unchanged.
    payload = json.dumps(
        [state, source_name, source_row, gmap_id, time_ms, text_sha256],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256_bytes(payload)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def score_chunk_files(score_dir: Path) -> list[Path]:
    files: list[tuple[int, Path]] = []
    unexpected = []
    for path in score_dir.glob("chunk_*.csv"):
        match = CHUNK_RE.fullmatch(path.name)
        if not match:
            unexpected.append(path.name)
        else:
            files.append((int(match.group(1)), path))
    if unexpected:
        raise IntegrityError(f"unexpected score chunk names: {unexpected[:10]}")
    files.sort(key=lambda item: item[0])
    observed = [number for number, _ in files]
    if not observed:
        raise IntegrityError(f"no chunk_*.csv files found in {score_dir}")
    if observed != list(range(len(observed))):
        raise IntegrityError(
            "score chunk numbers must be unique and contiguous from zero; "
            f"first observed={observed[:10]}, final={observed[-10:]}"
        )
    return [path for _, path in files]


def integral_series(series: pd.Series, label: str, source: Path) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    as_float = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(as_float).all():
        raise IntegrityError(f"{source}: {label} contains missing/non-finite values")
    if not np.equal(as_float, np.floor(as_float)).all():
        raise IntegrityError(f"{source}: {label} contains non-integral values")
    return pd.Series(as_float.astype(np.int64), index=series.index, name=label)


def read_and_validate_score_chunk(path: Path, year: int) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False, keep_default_na=False)
    if frame.columns.duplicated().any():
        raise IntegrityError(f"{path}: duplicate column names")
    expected = [*KEY_COLUMNS, *SCORE_COLUMNS]
    if list(frame.columns) != expected:
        missing = sorted(set(expected) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(expected))
        raise IntegrityError(
            f"{path}: score schema/order mismatch; missing={missing}; extra={extra}"
        )
    if frame.empty:
        raise IntegrityError(f"{path}: historical score chunk is unexpectedly empty")
    if frame["gmap_id"].eq("").any():
        raise IntegrityError(f"{path}: blank gmap_id")
    frame["gmap_id"] = frame["gmap_id"].astype(str)
    for name in ["year", "month", "week"]:
        frame[name] = integral_series(frame[name], name, path)
    if not frame["year"].eq(year).all():
        found = sorted(frame["year"].unique().tolist())[:20]
        raise IntegrityError(f"{path}: expected year {year}, found {found}")
    for name in SCORE_COLUMNS:
        numeric = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise IntegrityError(f"{path}: {name} contains missing/non-finite values")
        frame[name] = numeric.astype(np.float32)
    return frame


def compare_keys(raw_keys: pd.DataFrame, score: pd.DataFrame, score_path: Path) -> None:
    if len(raw_keys) != len(score):
        raise IntegrityError(
            f"{score_path}: row-count mismatch: replayed_raw={len(raw_keys)}, "
            f"score_rows={len(score)}"
        )
    score_keys = score[KEY_COLUMNS].reset_index(drop=True)
    raw_keys = raw_keys[KEY_COLUMNS].reset_index(drop=True)
    equality = pd.DataFrame(
        {
            name: raw_keys[name].to_numpy() == score_keys[name].to_numpy()
            for name in KEY_COLUMNS
        }
    )
    row_ok = equality.all(axis=1).to_numpy()
    if row_ok.all():
        return
    first = int(np.flatnonzero(~row_ok)[0])
    raise IntegrityError(
        f"{score_path}: rowwise key mismatch at row {first}; "
        f"raw={raw_keys.iloc[first].to_dict()}; "
        f"score={score_keys.iloc[first].to_dict()}"
    )


def validate_paths(args: argparse.Namespace, manifest_path: Path) -> None:
    if not 2007 <= args.year <= 2021:
        raise ValueError("--year must be between 2007 and 2021")
    if not args.raw_review_gz.is_file():
        raise FileNotFoundError(args.raw_review_gz)
    if not args.score_dir.is_dir():
        raise NotADirectoryError(args.score_dir)
    expected_raw_name = f"review-{args.state}.json.gz"
    if args.raw_review_gz.name != expected_raw_name:
        raise IntegrityError(
            f"raw filename must be {expected_raw_name!r}, got {args.raw_review_gz.name!r}"
        )
    expected_score_name = (
        f"temp_{args.state}" if args.year == 2019 else f"temp_{args.state}_{args.year}"
    )
    if args.score_dir.name != expected_score_name:
        raise IntegrityError(
            f"score directory for this state/year must be {expected_score_name!r}, "
            f"got {args.score_dir.name!r}"
        )
    if args.output_parquet.suffix != ".parquet":
        raise ValueError("--output-parquet must end in .parquet")
    if args.output_parquet.exists():
        raise FileExistsError(f"refusing to overwrite output: {args.output_parquet}")
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {manifest_path}")
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Opening catches cloud placeholders and permission failures before a long job.
    for source in [args.raw_review_gz]:
        try:
            with source.open("rb") as handle:
                if not handle.read(1):
                    raise IntegrityError(f"source is empty: {source}")
        except OSError as exc:
            raise IntegrityError(f"source is not locally readable: {source}: {exc}") from exc


def build_output_frame(
    *,
    args: argparse.Namespace,
    raw_chunk: pd.DataFrame,
    year_mask: pd.Series,
    eligible_mask_within_year: pd.Series,
    derived_year: pd.Series,
    derived_month: pd.Series,
    derived_week: pd.Series,
    raw_offset: int,
    raw_chunk_number: int,
    score_chunk_number: int,
    score: pd.DataFrame,
) -> pd.DataFrame:
    year_positions = np.flatnonzero(year_mask.to_numpy())
    chosen_positions = year_positions[eligible_mask_within_year.to_numpy()]
    chosen = raw_chunk.iloc[chosen_positions].copy()
    source_rows = raw_offset + chosen_positions

    if chosen["gmap_id"].isna().any():
        raise IntegrityError("eligible raw row has missing gmap_id")
    gmap = chosen["gmap_id"].astype(str).reset_index(drop=True)
    if gmap.eq("").any():
        raise IntegrityError("eligible raw row has blank gmap_id")
    time_ms = integral_series(chosen["time"].reset_index(drop=True), "time", args.raw_review_gz)

    raw_text = chosen["text"].fillna("").astype(str).reset_index(drop=True)
    clean_text = raw_text.str.replace(CLEAN_PATTERN, "", regex=True)
    text_hashes = raw_text.map(sha256_bytes)
    clean_hashes = clean_text.map(sha256_bytes)
    uids = [
        review_uid(
            args.state,
            args.raw_review_gz.name,
            int(source_row),
            str(place),
            int(timestamp),
            str(text_hash),
        )
        for source_row, place, timestamp, text_hash in zip(
            source_rows, gmap, time_ms, text_hashes, strict=True
        )
    ]

    output = pd.DataFrame(
        {
            "review_uid": uids,
            "state": args.state,
            "source_row_0based": source_rows.astype(np.int64),
            "raw_chunk_0based": np.full(len(chosen), raw_chunk_number, dtype=np.int32),
            "row_within_raw_chunk_0based": chosen_positions.astype(np.int32),
            "score_chunk_ordinal": np.full(len(chosen), score_chunk_number, dtype=np.int32),
            "gmap_id": gmap,
            "time_ms": time_ms,
            "year": derived_year.loc[year_mask].loc[eligible_mask_within_year].to_numpy(dtype=np.int16),
            "month": derived_month.loc[year_mask].loc[eligible_mask_within_year].to_numpy(dtype=np.int8),
            "week": derived_week.loc[year_mask].loc[eligible_mask_within_year].to_numpy(dtype=np.int8),
            "review_text": raw_text,
            "review_text_sha256": text_hashes,
            "scorer_clean_text_sha256": clean_hashes,
        }
    )
    for name in SCORE_COLUMNS:
        output[name] = score[name].to_numpy(dtype=np.float32)
    return output


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or Path(f"{args.output_parquet}.manifest.json")
    validate_paths(args, manifest_path)
    scores = score_chunk_files(args.score_dir)
    # Verify every score chunk is actually readable before replaying a large raw file.
    for path in scores:
        try:
            with path.open("rb") as handle:
                if not handle.read(1):
                    raise IntegrityError(f"score chunk is empty: {path}")
        except OSError as exc:
            raise IntegrityError(f"score chunk is not locally readable: {path}: {exc}") from exc

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{args.output_parquet.name}.",
        suffix=".partial",
        dir=args.output_parquet.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    temporary.unlink()

    writer: pq.ParquetWriter | None = None
    raw_offset = 0
    raw_rows_total = 0
    raw_rows_requested_year = 0
    cleaned_blank_requested_year = 0
    aligned_rows = 0
    score_rows = 0
    consumed_score_chunks = 0
    nonempty_replayed_raw_chunks = 0
    try:
        metadata = {
            b"alignment_protocol": PROTOCOL_VERSION.encode(),
            b"state": args.state.encode(),
            b"year": str(args.year).encode(),
            b"historical_raw_chunksize": str(HISTORICAL_RAW_CHUNKSIZE).encode(),
            b"clean_pattern": CLEAN_PATTERN.encode(),
        }
        schema = OUTPUT_SCHEMA.with_metadata(metadata)
        writer = pq.ParquetWriter(
            temporary,
            schema=schema,
            compression=args.compression,
            use_dictionary=True,
            write_statistics=True,
        )

        reader = pd.read_json(
            args.raw_review_gz,
            lines=True,
            compression="gzip",
            chunksize=HISTORICAL_RAW_CHUNKSIZE,
        )
        for raw_chunk_number, raw_chunk in enumerate(reader):
            missing_raw = {"gmap_id", "time", "text"} - set(raw_chunk.columns)
            if missing_raw:
                raise IntegrityError(
                    f"raw chunk {raw_chunk_number} lacks {sorted(missing_raw)}"
                )
            raw_rows_total += len(raw_chunk)

            # Deliberately identical to the historical script: no utc=True and no
            # errors='coerce'.  Integer Unix milliseconds derive the same fields.
            timestamp = pd.to_datetime(raw_chunk["time"], unit="ms")
            derived_year = timestamp.dt.year
            derived_month = timestamp.dt.month
            derived_week = timestamp.dt.isocalendar().week
            year_mask = derived_year.eq(args.year)
            raw_rows_requested_year += int(year_mask.sum())
            if not year_mask.any():
                raw_offset += len(raw_chunk)
                continue

            year_text = raw_chunk.loc[year_mask, "text"].fillna("").astype(str)
            cleaned = year_text.str.replace(CLEAN_PATTERN, "", regex=True)
            eligible = cleaned.str.strip().ne("")
            cleaned_blank_requested_year += int((~eligible).sum())
            if not eligible.any():
                raw_offset += len(raw_chunk)
                continue

            if consumed_score_chunks >= len(scores):
                raise IntegrityError(
                    "replay produced more nonempty chunks than historical scores; "
                    f"first missing score ordinal={consumed_score_chunks}"
                )
            score_path = scores[consumed_score_chunks]
            score = read_and_validate_score_chunk(score_path, args.year)

            chosen_index = year_text.index[eligible]
            raw_keys = pd.DataFrame(
                {
                    "gmap_id": raw_chunk.loc[chosen_index, "gmap_id"].astype(str).to_numpy(),
                    "year": derived_year.loc[chosen_index].to_numpy(dtype=np.int64),
                    "month": derived_month.loc[chosen_index].to_numpy(dtype=np.int64),
                    "week": derived_week.loc[chosen_index].to_numpy(dtype=np.int64),
                }
            )
            compare_keys(raw_keys, score, score_path)

            output = build_output_frame(
                args=args,
                raw_chunk=raw_chunk,
                year_mask=year_mask,
                eligible_mask_within_year=eligible,
                derived_year=derived_year,
                derived_month=derived_month,
                derived_week=derived_week,
                raw_offset=raw_offset,
                raw_chunk_number=raw_chunk_number,
                score_chunk_number=consumed_score_chunks,
                score=score,
            )
            table = pa.Table.from_pandas(
                output, schema=schema, preserve_index=False, safe=True
            )
            writer.write_table(table, row_group_size=HISTORICAL_RAW_CHUNKSIZE)

            rows = len(output)
            aligned_rows += rows
            score_rows += len(score)
            consumed_score_chunks += 1
            nonempty_replayed_raw_chunks += 1
            raw_offset += len(raw_chunk)

        if consumed_score_chunks != len(scores):
            raise IntegrityError(
                "historical score directory has unconsumed chunks: "
                f"consumed={consumed_score_chunks}, available={len(scores)}"
            )
        if aligned_rows == 0:
            raise IntegrityError("state/year produced zero aligned rows")
        if not (
            aligned_rows == score_rows
            and consumed_score_chunks == nonempty_replayed_raw_chunks
        ):
            raise IntegrityError("internal final count invariant failed")
        writer.close()
        writer = None
        os.replace(temporary, args.output_parquet)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise

    output_hash = sha256_file(args.output_parquet)
    score_inventory = [
        {"name": path.name, "size_bytes": path.stat().st_size} for path in scores
    ]
    inventory_hash = sha256_bytes(
        json.dumps(score_inventory, sort_keys=True, separators=(",", ":"))
    )
    manifest = {
        "status": "complete_state_year_exact_key_alignment",
        "protocol_version": PROTOCOL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": args.state,
        "year": args.year,
        "historical_replay": {
            "pandas_read_json_chunksize": HISTORICAL_RAW_CHUNKSIZE,
            "date_derivation": "pd.to_datetime(time, unit='ms'); year/month/isocalendar.week",
            "operation_order": "derive date -> filter year -> fillna/cast text -> regex clean -> strip nonblank",
            "clean_pattern": CLEAN_PATTERN,
            "score_chunk_mapping": "one dense score ordinal per nonempty replayed raw chunk",
            "rowwise_required_equal_keys": KEY_COLUMNS,
            "residual_identifiability_limit": (
                "historical score rows omit text and a stable review ID; passing all "
                "recoverable keys is the strongest available reconstruction check, "
                "not cryptographic proof among consecutive rows sharing every key"
            ),
        },
        "source": {
            "raw_review_gz": str(args.raw_review_gz.resolve()),
            "raw_size_bytes": args.raw_review_gz.stat().st_size,
            "score_dir": str(args.score_dir.resolve()),
            "score_chunk_count": len(scores),
            "score_chunk_inventory_sha256": inventory_hash,
            "score_total_size_bytes": sum(item["size_bytes"] for item in score_inventory),
        },
        "counts": {
            "raw_rows_scanned": raw_rows_total,
            "raw_rows_requested_year_before_text_filter": raw_rows_requested_year,
            "cleaned_blank_rows_requested_year": cleaned_blank_requested_year,
            "aligned_rows": aligned_rows,
            "historical_score_rows": score_rows,
            "nonempty_replayed_raw_chunks": nonempty_replayed_raw_chunks,
            "consumed_score_chunks": consumed_score_chunks,
        },
        "output": {
            "path": str(args.output_parquet.resolve()),
            "sha256": output_hash,
            "size_bytes": args.output_parquet.stat().st_size,
            "compression": args.compression,
            "columns": OUTPUT_SCHEMA.names,
            "score_dimensions": len(SCORE_COLUMNS),
        },
        "runtime": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
        "deliberately_not_computed": {
            "duplicate_review_counts": {
                "status": "REQUIRED_LATER_BEFORE_PLACE_AGGREGATION",
                "reason": "review_uid is a source-occurrence ID; semantic/exact duplicate policy is not yet frozen",
            },
            "place_type_mapping": {
                "status": "REQUIRED_LATER_FOR_STRATIFIED_QA",
                "reason": "no place-type map is used or assumed during alignment",
            },
            "protected_holdout_outcomes": "NEVER_ACCESSED",
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (IntegrityError, ValueError, KeyError, OSError, pd.errors.ParserError) as exc:
        print(f"FATAL ALIGNMENT ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
