#!/usr/bin/env python3
import copy,json,os,tempfile
from pathlib import Path
import score_multitarget_v1_1 as cli_base
import score_multitarget_v1_2_6 as previous
IntegrityError=cli_base.IntegrityError;TARGETS=cli_base.TARGETS;sha=cli_base.sha;load_json=cli_base.load_json;score_frame=previous.score_frame
def validate_artifact(path,expected):
    path=Path(path)
    if not path.is_file() or path.is_symlink() or sha(path)!=expected:raise IntegrityError("model artifact hash mismatch")
    a=load_json(path);names=("score_multitarget_v1_2_7.py","score_multitarget_v1_2_6.py","score_multitarget_v1_2_5.py","score_multitarget_v1_2_4.py","score_multitarget_v1_2_3.py","score_multitarget_v1_2_2.py","score_multitarget_v1_1.py","text_context_features_v1_1.py","t05_structural_features_v1_2.py");actual={n:sha(Path(__file__).with_name(n)) for n in names}
    if a.get("schema_version")!="dimension29-operational-linear-v1.2.7" or a.get("required_program_sha256")!=actual:raise IntegrityError("artifact schema/program hash")
    shadow=copy.deepcopy(a);shadow["schema_version"]="dimension29-operational-linear-v1.2.6";shadow["required_program_sha256"]={n:actual[n] for n in names[1:]};fd,tmp=tempfile.mkstemp(prefix=".artifact-v1.2.7-validation.",suffix=".json");os.close(fd)
    try:Path(tmp).write_text(json.dumps(shadow,sort_keys=True)+"\n");previous.validate_artifact(tmp,sha(tmp))
    finally:Path(tmp).unlink(missing_ok=True)
    return a
if __name__=="__main__":cli_base.validate_artifact=validate_artifact;cli_base.score_frame=score_frame;cli_base.main()
