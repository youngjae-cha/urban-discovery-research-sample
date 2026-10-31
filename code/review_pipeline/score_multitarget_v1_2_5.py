#!/usr/bin/env python3
"""v1.2.5 scorer contract; fitted review-level predictions are unchanged."""
import copy, json, os, tempfile
from pathlib import Path
import score_multitarget_v1_1 as cli_base
import score_multitarget_v1_2_4 as previous

IntegrityError=cli_base.IntegrityError; TARGETS=cli_base.TARGETS; sha=cli_base.sha; load_json=cli_base.load_json; score_frame=previous.score_frame

def validate_artifact(path,expected):
    path=Path(path)
    if not path.is_file() or path.is_symlink() or sha(path)!=expected: raise IntegrityError("model artifact hash mismatch")
    artifact=load_json(path)
    names=("score_multitarget_v1_2_5.py","score_multitarget_v1_2_4.py","score_multitarget_v1_2_3.py","score_multitarget_v1_2_2.py","score_multitarget_v1_1.py","text_context_features_v1_1.py","t05_structural_features_v1_2.py")
    actual={name:sha(Path(__file__).with_name(name)) for name in names}
    if artifact.get("schema_version")!="dimension29-operational-linear-v1.2.5" or artifact.get("required_program_sha256")!=actual: raise IntegrityError("artifact schema/program hash")
    shadow=copy.deepcopy(artifact); shadow["schema_version"]="dimension29-operational-linear-v1.2.4"; shadow["required_program_sha256"]={name:actual[name] for name in names[1:]}
    fd,temp=tempfile.mkstemp(prefix=".artifact-v1.2.5-validation.",suffix=".json"); os.close(fd)
    try:
        Path(temp).write_text(json.dumps(shadow,sort_keys=True)+"\n"); previous.validate_artifact(temp,sha(temp))
    finally: Path(temp).unlink(missing_ok=True)
    return artifact

if __name__=="__main__":
    cli_base.validate_artifact=validate_artifact; cli_base.score_frame=score_frame; cli_base.main()
