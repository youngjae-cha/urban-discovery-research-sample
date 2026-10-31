#!/usr/bin/env python3
"""Strict v1.2.2 scorer for the six operational-provisional evidence models."""
from pathlib import Path
import math

import numpy as np
import pandas as pd

import score_multitarget_v1_1 as cli_base
from text_context_features_v1_1 import OUTPUT_COLUMNS, extract_context
from t05_structural_features_v1_2 import OUTPUT_COLUMNS as T05_COLUMNS, extract_t05

IntegrityError = cli_base.IntegrityError
TARGETS = cli_base.TARGETS
sha = cli_base.sha
load_json = cli_base.load_json

EXPECTED_TOP_KEYS = {
    "algorithm", "artifact_id", "built_utc", "claim_ceiling", "context_features",
    "evidence_targets", "excluded_from_evidence_models", "rank_policy",
    "raw_score_columns", "required_program_sha256", "schema_version",
    "source_provenance", "status", "t11_policy",
}
EXPECTED_MODEL_KEYS = {
    "candidate_id", "coefficients", "feature_order", "fixed_probability_cutoff",
    "imputation_medians", "intercept", "standard_scaler_mean",
    "standard_scaler_scale", "target_id", "training",
}
EXPECTED_TRAINING_KEYS = {
    "assignment_ids_sha256", "negatives", "places", "positives", "rows",
    "training_rows_sha256",
}
EXPECTED_PROVENANCE_KEYS = {
    "candidate_spec_sha256", "eligible_only", "immediate_parent_artifact_sha256",
    "modeling_v1_1_table_sha256", "t05_corrected_report_sha256",
    "t05_independent_audit_sha256", "t05_lock_sha256", "t05_source_sha256",
    "t05_structural_table_sha256", "t05_v1_1_manifest_sha256",
}
EXPECTED_RAW = [
    "Appreciation_Score", "Coherence_Score", "Cosmopolitanism_Score",
    "Curiosity_Embracing_Score", "Curiosity_Stretching_Score",
    "Exploration_Behavior_Score", "FG_Richness_Score", "Happiness_Score",
    "Intolerance_of_Uncertainty_Score", "Learning_Knowledge_Score",
    "Mattering_Score", "Meaningfulness_MLQ_Score", "Meaningfulness_Score",
    "Need_for_Cognition_Score", "Openness_to_Uncertainty_Score",
    "PR_Scale_Overall_Score", "Perspective_Change_Score", "Price_Benchmark_Score",
    "Psychological_Richness_Score", "Purpose_MLQ_Score", "SPANE_NA_Score",
    "SPANE_PA_Score", "Satisfaction_SWLS_Score", "Self_Expansion_Score",
    "Service_Speed_Benchmark_Score", "Shared_Reality_Score", "Surprise_Score",
    "Transformative_Experience_Score", "Vitality_Aliveness_Score",
]
EXPECTED_T05_FEATURES = [
    "Mattering_Score", "t05_duration_recurrence_category_count",
    "t05_relationship_category_count", "t05_autobiographical_past_category_count",
    "Service_Speed_Benchmark_Score",
]
EXPECTED_T05_PROVENANCE = {
    "t05_corrected_report_sha256": "494725cb309335414380d76ee0500bd1aac925a2aa7fb2898a20143578c43cca",
    "t05_independent_audit_sha256": "045c0ee7e3ea9bf8b98fa2526c0f1ad1dfa5bc6383345bb89a89535bae92f2f4",
    "t05_v1_1_manifest_sha256": "047ed359eb744d3d78b2238fa4d034a9b48499ab50e422a5a57b40635bebd6ba",
}
EXPECTED_TRAINING = {
    "T01_positive_affect_enjoyment": (55, 15, 40, 38),
    "T02_visit_satisfaction": (55, 31, 24, 39),
    "T05_autobiographical_attachment_salience": (55, 10, 45, 32),
    "T08_experiential_variety_complexity": (54, 14, 40, 41),
    "T09_exploration_discovery": (56, 10, 46, 41),
    "T15_learning_knowledge_gain": (56, 17, 39, 38),
}


def validate_artifact(path, expected):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or sha(path) != expected:
        raise IntegrityError("model artifact hash mismatch")
    artifact = load_json(path)
    if set(artifact) != EXPECTED_TOP_KEYS:
        raise IntegrityError("artifact top-level schema")
    if artifact["schema_version"] != "dimension29-operational-linear-v1.2.2" or artifact["status"] != cli_base.STATUS:
        raise IntegrityError("artifact schema/status")
    if artifact["claim_ceiling"] != "AI_SUPERVISED_DEVELOPMENT_OPERATIONAL_PROVISIONAL_NOT_HUMAN_VALIDITY_OR_POPULATION_PREVALENCE":
        raise IntegrityError("claim ceiling")
    if artifact["raw_score_columns"] != EXPECTED_RAW or artifact["context_features"] != OUTPUT_COLUMNS:
        raise IntegrityError("raw29/context order")
    if artifact["excluded_from_evidence_models"] != ["T03_personal_meaning_significance", "T11_transformative_perspective_change"]:
        raise IntegrityError("excluded target contract")
    if artifact["t11_policy"] != "raw cosines only; no evidence model" or artifact["rank_policy"] is not None:
        raise IntegrityError("T11/rank contract")
    actual_programs = {
        "score_multitarget_v1_2_2.py": sha(Path(__file__).resolve()),
        "score_multitarget_v1_1.py": sha(Path(__file__).with_name("score_multitarget_v1_1.py")),
        "text_context_features_v1_1.py": sha(Path(__file__).with_name("text_context_features_v1_1.py")),
        "t05_structural_features_v1_2.py": sha(Path(__file__).with_name("t05_structural_features_v1_2.py")),
    }
    if artifact["required_program_sha256"] != actual_programs:
        raise IntegrityError("program hash mismatch")
    provenance = artifact["source_provenance"]
    if set(provenance) != EXPECTED_PROVENANCE_KEYS or provenance.get("eligible_only") is not True:
        raise IntegrityError("source provenance schema")
    if any(provenance.get(k) != v for k, v in EXPECTED_T05_PROVENANCE.items()):
        raise IntegrityError("corrected T05 provenance hash")
    algorithm = artifact["algorithm"]
    if set(algorithm) != {"class", "parameters", "sklearn_version"} or algorithm["class"] != "sklearn.linear_model.LogisticRegression":
        raise IntegrityError("algorithm schema")
    if algorithm["parameters"] != {"C": .25, "class_weight": "balanced", "max_iter": 2000, "penalty": "l2", "random_state": 20260903, "solver": "liblinear"}:
        raise IntegrityError("algorithm parameters")
    if [m.get("target_id") for m in artifact["evidence_targets"]] != TARGETS:
        raise IntegrityError("target inventory/order")
    for model in artifact["evidence_targets"]:
        if set(model) != EXPECTED_MODEL_KEYS or set(model.get("training", {})) != EXPECTED_TRAINING_KEYS:
            raise IntegrityError("model/training schema")
        target = model["target_id"]
        expected_features = EXPECTED_T05_FEATURES if target.startswith("T05_") else EXPECTED_RAW + OUTPUT_COLUMNS
        if model["feature_order"] != expected_features:
            raise IntegrityError(f"exact feature order {target}")
        if target.startswith("T05_"):
            if model["candidate_id"] != "T05_reduced_A_no_spane_pa_v1":
                raise IntegrityError("T05 candidate")
        elif model["candidate_id"] != f"{target}__logistic_l2_all29_context_v1":
            raise IntegrityError(f"candidate id {target}")
        n = len(expected_features)
        for key in ("imputation_medians", "standard_scaler_mean", "standard_scaler_scale", "coefficients"):
            vector = np.asarray(model[key], dtype=float)
            if vector.shape != (n,) or not np.isfinite(vector).all():
                raise IntegrityError(f"model vector {key}")
        if np.any(np.asarray(model["standard_scaler_scale"], dtype=float) <= 0):
            raise IntegrityError("nonpositive scaler")
        if not math.isfinite(float(model["intercept"])) or model["fixed_probability_cutoff"] != .5:
            raise IntegrityError("intercept/cutoff")
        training = model["training"]
        observed = (training["rows"], training["positives"], training["negatives"], training["places"])
        if observed != EXPECTED_TRAINING[target]:
            raise IntegrityError(f"training counts {target}")
        if not all(isinstance(training[k], str) and len(training[k]) == 64 for k in ("assignment_ids_sha256", "training_rows_sha256")):
            raise IntegrityError("training hashes")
    return artifact


def score_frame(frame, artifact):
    required = ["review_uid", "gmap_id", "review_text", "normalized_text_hash",
                "normalized_duplicate_count_within_place", "n_words", "generic_only_flag",
                *artifact["raw_score_columns"]]
    missing = sorted(set(required) - set(frame))
    if missing:
        raise IntegrityError(f"missing input features: {missing}")
    raw = frame[artifact["raw_score_columns"]]
    if not all(pd.api.types.is_numeric_dtype(raw[c]) for c in raw) or not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise IntegrityError("raw scores")
    context = pd.DataFrame([extract_context(x) for x in frame.review_text], index=frame.index)
    t05 = pd.DataFrame([extract_t05(x) for x in frame.review_text], index=frame.index)
    if not np.array_equal(frame.generic_only_flag.astype(bool).to_numpy(), context.ctx_generic_only_flag.astype(bool).to_numpy()):
        raise IntegrityError("generic_only lineage")
    n_words = pd.to_numeric(frame.n_words, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(n_words).all() or np.any(n_words < 0) or not np.all(n_words == np.floor(n_words)):
        raise IntegrityError("n_words must be finite nonnegative integers")
    duplicates = pd.to_numeric(frame.normalized_duplicate_count_within_place, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(duplicates).all() or np.any(duplicates < 1) or not np.all(duplicates == np.floor(duplicates)):
        raise IntegrityError("duplicate counts")
    keep = [c for c in ["review_uid", "gmap_id", "normalized_text_hash", "state", "time_ms", "year", "month", "week", "n_words", "generic_only_flag", "normalized_duplicate_count_within_place"] if c in frame]
    out = frame[keep].copy()
    out["within_place_duplicate_weight"] = 1.0 / duplicates
    out["common_eligible_substantive"] = (~context.ctx_generic_only_flag.astype(bool).to_numpy()) & (n_words > 4)
    for column in OUTPUT_COLUMNS:
        out[column] = context[column].astype(float)
    for column in T05_COLUMNS:
        out[column] = t05[column].astype(np.int16)
    for column in artifact["raw_score_columns"]:
        out[column] = frame[column]
    features = pd.concat([frame[artifact["raw_score_columns"]], context, t05], axis=1)
    for model in artifact["evidence_targets"]:
        # copy=True is required under pandas copy-on-write (including pandas 3.x).
        x = features[model["feature_order"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
        if np.isinf(x).any():
            raise IntegrityError("infinite model feature")
        missing_values = np.isnan(x)
        if missing_values.any():
            x[missing_values] = np.broadcast_to(np.asarray(model["imputation_medians"], dtype=float), x.shape)[missing_values]
        z = ((x - np.asarray(model["standard_scaler_mean"], dtype=float)) / np.asarray(model["standard_scaler_scale"], dtype=float)) @ np.asarray(model["coefficients"], dtype=float) + float(model["intercept"])
        if not np.isfinite(z).all():
            raise IntegrityError("nonfinite predictor")
        probability = cli_base.sigmoid(z)
        target = model["target_id"]
        out[target + "__evidence_probability"] = probability
        out[target + "__fixed_p05_call"] = probability >= .5
    if not np.array_equal(out[artifact["raw_score_columns"]].to_numpy(), frame[artifact["raw_score_columns"]].to_numpy()):
        raise IntegrityError("raw29 changed")
    return out


if __name__ == "__main__":
    cli_base.validate_artifact = validate_artifact
    cli_base.score_frame = score_frame
    cli_base.main()
