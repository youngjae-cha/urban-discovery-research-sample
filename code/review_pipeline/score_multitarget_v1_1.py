#!/usr/bin/env python3
"""Hash-bound six-target operational-provisional Parquet scorer."""
from __future__ import annotations
import argparse, hashlib, json, math, os, tempfile
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
from text_context_features_v1_1 import OUTPUT_COLUMNS, extract_context

STATUS="DEVELOPMENT_SELECTED_OPERATIONAL_PROVISIONAL"
TARGETS=["T01_positive_affect_enjoyment","T02_visit_satisfaction","T05_autobiographical_attachment_salience","T08_experiential_variety_complexity","T09_exploration_discovery","T15_learning_knowledge_gain"]
PROTECTED=("b40","b60","locked-23","locked_23","protected","heldout","held-out","holdout")
class IntegrityError(RuntimeError): pass
def sha(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def load_json(p):
    def hook(xs):
        d={}
        for k,v in xs:
            if k in d: raise IntegrityError(f"duplicate JSON key {k}")
            d[k]=v
        return d
    with open(p,encoding="utf-8") as handle:
        return json.load(handle,object_pairs_hook=hook,parse_constant=lambda x:(_ for _ in()).throw(IntegrityError(f"nonfinite {x}")))
def check_safe(*paths):
    for p in paths:
        low=str(Path(p).resolve()).lower()
        if any(t in low for t in PROTECTED): raise IntegrityError(f"protected path rejected: {p}")
def validate_artifact(path,expected):
    if not Path(path).is_file() or Path(path).is_symlink() or sha(path)!=expected: raise IntegrityError("model artifact hash mismatch")
    a=load_json(path)
    keys={"algorithm","artifact_id","built_utc","claim_ceiling","context_features","evidence_targets","excluded_from_evidence_models","rank_policy","raw_score_columns","required_program_sha256","schema_version","source_provenance","status","t11_policy"}
    if set(a)!=keys or a["schema_version"]!="dimension29-operational-linear-v1.1" or a["status"]!=STATUS: raise IntegrityError("artifact schema/status mismatch")
    if a["claim_ceiling"]!="AI_SUPERVISED_DEVELOPMENT_OPERATIONAL_PROVISIONAL_NOT_HUMAN_VALIDITY_OR_POPULATION_PREVALENCE": raise IntegrityError("claim ceiling")
    if len(a["raw_score_columns"])!=29 or len(set(a["raw_score_columns"]))!=29: raise IntegrityError("raw29 contract")
    if a["context_features"]!=OUTPUT_COLUMNS or a["rank_policy"] is not None: raise IntegrityError("context/rank policy")
    if a["excluded_from_evidence_models"]!=["T03_personal_meaning_significance","T11_transformative_perspective_change"]: raise IntegrityError("excluded target contract")
    programs=a["required_program_sha256"]
    actual={"score_multitarget_v1_1.py":sha(Path(__file__).resolve()),"text_context_features_v1_1.py":sha(Path(__file__).with_name("text_context_features_v1_1.py"))}
    if programs!=actual: raise IntegrityError("program hash mismatch")
    if [m.get("target_id") for m in a["evidence_targets"]]!=TARGETS: raise IntegrityError("target inventory/order")
    params=a["algorithm"].get("parameters",{})
    if params!={"C":.25,"class_weight":"balanced","max_iter":2000,"penalty":"l2","random_state":20260903,"solver":"liblinear"}: raise IntegrityError("algorithm parameters")
    for m in a["evidence_targets"]:
        n=len(m.get("feature_order",[]))
        if n!=35 or len(set(m["feature_order"]))!=35: raise IntegrityError("feature order")
        for k in ("imputation_medians","standard_scaler_mean","standard_scaler_scale","coefficients"):
            if len(m.get(k,[]))!=n or not np.isfinite(np.asarray(m[k],float)).all(): raise IntegrityError(f"model vector {k}")
        if np.any(np.asarray(m["standard_scaler_scale"],float)<=0) or not math.isfinite(float(m["intercept"])) or m["fixed_probability_cutoff"]!=.5: raise IntegrityError("scale/intercept/cutoff")
        tr=m.get("training",{})
        if tr.get("rows")!=tr.get("positives",0)+tr.get("negatives",0) or tr.get("rows",0)<=0: raise IntegrityError("training counts")
        if not all(isinstance(tr.get(k),str) and len(tr[k])==64 for k in ("assignment_ids_sha256","training_rows_sha256")): raise IntegrityError("training hashes")
    return a
def validate_manifest(path,input_path,input_hash):
    m=load_json(path); g=m.get("global",{})
    if m.get("status")!="passed" or m.get("mode")!="production" or not g.get("full_content_scan_complete") or not g.get("exact_global_uid_check") or not g.get("duplicate_weighting_ready"): raise IntegrityError("production manifest global gate")
    matches=[x for x in m.get("shards",[]) if Path(str(x.get("path",""))).resolve()==Path(input_path).resolve()]
    if len(matches)!=1 or matches[0].get("sha256")!=input_hash or matches[0].get("status")!="passed": raise IntegrityError("input not exactly bound by production manifest")
    return sha(path)
def sigmoid(z):
    out=np.empty_like(z,dtype=np.float64); pos=z>=0; out[pos]=1/(1+np.exp(-z[pos])); e=np.exp(z[~pos]); out[~pos]=e/(1+e); return out
def score_frame(frame,a):
    required=["review_uid","gmap_id","review_text","normalized_text_hash","normalized_duplicate_count_within_place","n_words","generic_only_flag",*a["raw_score_columns"]]
    miss=sorted(set(required)-set(frame));
    if miss: raise IntegrityError(f"missing input features: {miss}")
    raw=frame[a["raw_score_columns"]]
    if not all(pd.api.types.is_numeric_dtype(raw[c]) for c in raw) or not np.isfinite(raw.to_numpy(float)).all(): raise IntegrityError("raw scores nonnumeric/nonfinite")
    contexts=pd.DataFrame([extract_context(x) for x in frame.review_text],index=frame.index)
    stored=frame.generic_only_flag.astype(bool).to_numpy(); derived=contexts.ctx_generic_only_flag.astype(bool).to_numpy()
    if not np.array_equal(stored,derived): raise IntegrityError("stored generic_only differs from frozen feature generator")
    dup=pd.to_numeric(frame.normalized_duplicate_count_within_place,errors="coerce").to_numpy(float)
    if not np.isfinite(dup).all() or np.any(dup<1) or not np.all(dup==np.floor(dup)): raise IntegrityError("duplicate counts")
    keep=[c for c in ["review_uid","gmap_id","normalized_text_hash","state","time_ms","year","month","week","n_words","generic_only_flag","normalized_duplicate_count_within_place"] if c in frame]
    out=frame[keep].copy(); out["within_place_duplicate_weight"]=1.0/dup; out["common_eligible_substantive"]=(~derived)&(pd.to_numeric(frame.n_words,errors="coerce").to_numpy(float)>4)
    for c in OUTPUT_COLUMNS: out[c]=contexts[c].astype(np.float64)
    for c in a["raw_score_columns"]: out[c]=frame[c]
    feature=pd.concat([frame[a["raw_score_columns"]],contexts],axis=1)
    for m in a["evidence_targets"]:
        X=feature[m["feature_order"]].apply(pd.to_numeric,errors="coerce").to_numpy(float); inf=np.isinf(X)
        if inf.any(): raise IntegrityError("infinite model feature")
        missing=np.isnan(X)
        if missing.any(): X[missing]=np.broadcast_to(np.asarray(m["imputation_medians"],float),X.shape)[missing]
        z=((X-np.asarray(m["standard_scaler_mean"],float))/np.asarray(m["standard_scaler_scale"],float))@np.asarray(m["coefficients"],float)+float(m["intercept"])
        if not np.isfinite(z).all(): raise IntegrityError("nonfinite linear predictor")
        p=sigmoid(z); tid=m["target_id"]; out[f"{tid}__evidence_probability"]=p; out[f"{tid}__fixed_p05_call"]=(p>=.5)
    if not np.array_equal(out[a["raw_score_columns"]].to_numpy(),frame[a["raw_score_columns"]].to_numpy()): raise IntegrityError("raw29 roundtrip changed")
    return out
def main():
    p=argparse.ArgumentParser(); p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--model-artifact",type=Path,required=True);p.add_argument("--model-artifact-sha256",required=True);p.add_argument("--production-manifest",type=Path);a=p.parse_args()
    check_safe(a.input,a.output,a.model_artifact,*([] if a.production_manifest is None else [a.production_manifest])); art=validate_artifact(a.model_artifact,a.model_artifact_sha256)
    if a.output.exists() or Path(str(a.output)+".receipt.json").exists(): raise FileExistsError("refuse overwrite")
    input_hash=sha(a.input); manifest_hash=validate_manifest(a.production_manifest,a.input,input_hash) if a.production_manifest else None
    table=pq.read_table(a.input); source=table.to_pandas(); scored=score_frame(source,art)
    a.output.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{a.output.name}.",dir=a.output.parent);os.close(fd);Path(tmp).unlink()
    try: pq.write_table(pa.Table.from_pandas(scored,preserve_index=False),tmp,compression="zstd");os.replace(tmp,a.output)
    finally: Path(tmp).unlink(missing_ok=True)
    receipt={"status":STATUS,"rows":len(scored),"input":str(a.input.resolve()),"input_sha256":input_hash,"output":str(a.output.resolve()),"output_sha256":sha(a.output),"model_artifact_sha256":sha(a.model_artifact),"production_manifest_sha256":manifest_hash,"raw_score_count":29,"raw_score_roundtrip_exact":True,"probability_dtype":"float64","rank_fields_emitted":False,"claim_ceiling":art["claim_ceiling"]}
    rp=Path(str(a.output)+".receipt.json");rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n");print(json.dumps(receipt,indent=2))
if __name__=="__main__": main()
