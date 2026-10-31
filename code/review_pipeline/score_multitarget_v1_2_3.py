#!/usr/bin/env python3
"""v1.2.3 scorer contract; review-level scoring is unchanged from strict v1.2.2."""
import copy
import json
import os
import tempfile
from pathlib import Path

import score_multitarget_v1_1 as cli_base
import score_multitarget_v1_2_2 as strict_base

IntegrityError = cli_base.IntegrityError
TARGETS = cli_base.TARGETS
sha = cli_base.sha
load_json = cli_base.load_json
score_frame = strict_base.score_frame


def validate_artifact(path, expected):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or sha(path) != expected:
        raise IntegrityError("model artifact hash mismatch")
    artifact = load_json(path)
    actual_programs = {
        "score_multitarget_v1_2_3.py": sha(Path(__file__).resolve()),
        "score_multitarget_v1_2_2.py": sha(Path(__file__).with_name("score_multitarget_v1_2_2.py")),
        "score_multitarget_v1_1.py": sha(Path(__file__).with_name("score_multitarget_v1_1.py")),
        "text_context_features_v1_1.py": sha(Path(__file__).with_name("text_context_features_v1_1.py")),
        "t05_structural_features_v1_2.py": sha(Path(__file__).with_name("t05_structural_features_v1_2.py")),
    }
    if artifact.get("schema_version") != "dimension29-operational-linear-v1.2.3":
        raise IntegrityError("artifact schema version")
    if artifact.get("required_program_sha256") != actual_programs:
        raise IntegrityError("program hash mismatch")
    # Reuse every strict v1.2.2 structural check on an in-memory-equivalent,
    # temporary representation. The released artifact itself remains immutable.
    shadow = copy.deepcopy(artifact)
    shadow["schema_version"] = "dimension29-operational-linear-v1.2.2"
    shadow["required_program_sha256"] = {
        "score_multitarget_v1_2_2.py": actual_programs["score_multitarget_v1_2_2.py"],
        "score_multitarget_v1_1.py": actual_programs["score_multitarget_v1_1.py"],
        "text_context_features_v1_1.py": actual_programs["text_context_features_v1_1.py"],
        "t05_structural_features_v1_2.py": actual_programs["t05_structural_features_v1_2.py"],
    }
    descriptor, temporary = tempfile.mkstemp(prefix=".artifact-v1.2.3-validation.", suffix=".json")
    os.close(descriptor)
    try:
        Path(temporary).write_text(json.dumps(shadow, sort_keys=True) + "\n")
        strict_base.validate_artifact(temporary, sha(temporary))
    finally:
        Path(temporary).unlink(missing_ok=True)
    return artifact


if __name__ == "__main__":
    cli_base.validate_artifact = validate_artifact
    cli_base.score_frame = score_frame
    cli_base.main()
