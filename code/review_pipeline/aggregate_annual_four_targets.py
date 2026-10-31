"""Annual four-target projection of frozen v1.2.8 aggregation arithmetic.

Stream each complete hash bucket independently, then merge additive place totals.
No changes to eligibility, duplicate weights, probabilities, or p=.5 calls.
No metadata/cohort filtering. This produces ALL observed place scores; metadata
attachment and final coverage checks are a separate necessary delivery step.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import duckdb
import pyarrow.parquet as pq

RAW = {"T09":"Exploration_Behavior_Score", "T15":"Learning_Knowledge_Score",
       "T08":"FG_Richness_Score", "T02":"Satisfaction_SWLS_Score"}
TARGET = {"T09":"T09_exploration_discovery", "T15":"T15_learning_knowledge_gain",
          "T08":"T08_experiential_variety_complexity", "T02":"T02_visit_satisfaction"}
STATUS="DEVELOPMENT_SELECTED_OPERATIONAL_PROVISIONAL"


def q(x): return '"'+str(x).replace('"','""')+'"'
def s(x): return "'"+str(x).replace("'","''")+"'"
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''): h.update(b)
    return h.hexdigest()


def write_json(path,x):
    with path.open('x') as f: json.dump(x,f,indent=2,allow_nan=False); f.write('\n')


def cluster_sql():
    w='within_place_duplicate_weight'
    e='CAST(common_eligible_substantive AS INTEGER)'
    expressions=["gmap_id","normalized_text_hash","count(*) AS raw_rows",
                 f"sum({e}) AS eligible_rows",f"sum({w}) AS all_mass",f"sum({w}*{e}) AS eligible_mass",
                 "min(normalized_duplicate_count_within_place) AS declared_min",
                 "max(normalized_duplicate_count_within_place) AS declared_max",
                 f"min({e}) AS eligibility_min",f"max({e}) AS eligibility_max"]
    for tid in RAW:
        expressions += [f"sum({w}*{q(RAW[tid])}) AS {q('L0_'+tid+'_sum')}",
                        f"sum({w}*{e}*{q(RAW[tid])}) AS {q('L1_'+tid+'_sum')}",
                        f"sum({w}*{e}*{q(TARGET[tid]+'__evidence_probability')}) AS {q('L2_'+tid+'_sum')}",
                        f"sum({w}*{e}*CAST({q(TARGET[tid]+'__fixed_p05_call')} AS DOUBLE)) AS {q('L3_'+tid+'_sum')}"]
    return "SELECT "+','.join(expressions)+" FROM r GROUP BY gmap_id,normalized_text_hash"


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scored-root',type=Path,required=True)
    ap.add_argument('--part-glob',default='part-*.parquet')
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--year',type=int,required=True)
    ap.add_argument('--expected-parts',type=int,required=True)
    ap.add_argument('--expected-rows',type=int,required=True)
    ap.add_argument('--model-artifact',type=Path,required=True)
    ap.add_argument('--model-sha256',required=True)
    ap.add_argument('--production-manifest',type=Path,required=True)
    ap.add_argument('--sealed-enrichment-receipt',type=Path,required=True)
    ap.add_argument('--spec-file',type=Path,required=True)
    ap.add_argument('--memory-limit',default='10GB')
    ap.add_argument('--threads',type=int,default=2)
    a=ap.parse_args(); started=time.time()
    assert '수정분 6' in a.spec_file.read_text()
    assert sha(a.model_artifact)==a.model_sha256
    artifact=json.loads(a.model_artifact.read_text())
    manifest_sha=sha(a.production_manifest)
    manifest=json.loads(a.production_manifest.read_text())
    sealed=json.loads(a.sealed_enrichment_receipt.read_text())
    assert manifest['status']=='passed' and manifest['mode']=='production' and manifest['year']==a.year
    assert manifest['global_scan']==dict(rows=a.expected_rows,unique_review_uids=a.expected_rows,invalid_rows=0)
    assert all(manifest['global'][k] is True for k in ['full_content_scan_complete','exact_global_uid_check','duplicate_weighting_ready'])
    assert manifest['model_artifact_sha256']==a.model_sha256
    assert manifest['sealed_enrichment_receipt_sha256']==sha(a.sealed_enrichment_receipt)
    assert sealed['source_count']==52 and sealed['rows']==a.expected_rows
    assert sealed['observed_bucket_count']==a.expected_parts
    assert sealed['status']=='PASS_READY_FOR_FROZEN_EVIDENCE_SCORING_ENGINEERING_ONLY'
    substrate={str(Path(x['path']).resolve()):x for x in manifest['shards']}
    assert len(substrate)==len(manifest['shards'])==a.expected_parts
    raw29=artifact['raw_score_columns']
    assert len(raw29)==29 and set(RAW.values()).issubset(raw29)
    parts=sorted(a.scored_root.glob(a.part_glob))
    assert len(parts)==a.expected_parts and len({p.resolve() for p in parts})==len(parts)
    assert not a.output_dir.exists()
    a.output_dir.mkdir(parents=True)
    partial_dir=a.output_dir/'partials'; partial_dir.mkdir()
    database=a.output_dir/'aggregation_work.duckdb'
    con=duckdb.connect(str(database))
    con.execute(f'SET memory_limit={s(a.memory_limit)}')
    con.execute(f'SET threads={a.threads}')
    con.execute('SET preserve_insertion_order=false')
    inventory=[]; total_rows=0; conflict_clusters=0; seen_inputs=set()
    sums=[f'{level}_{tid}_sum' for tid in RAW for level in ['L0','L1','L2','L3']]
    for part in parts:
        receipt_path=Path(str(part)+'.receipt.json')
        receipt=json.loads(receipt_path.read_text())
        digest=sha(part)
        assert receipt['status']==STATUS and receipt['output_sha256']==digest
        assert receipt['model_artifact_sha256']==a.model_sha256 and receipt['raw_score_roundtrip_exact'] is True
        assert receipt['production_manifest_sha256']==manifest_sha
        source=str(Path(receipt['input']).resolve())
        assert source not in seen_inputs and source in substrate
        assert substrate[source]['status']=='passed' and substrate[source]['sha256']==receipt['input_sha256']
        seen_inputs.add(source)
        con.execute(f'CREATE OR REPLACE VIEW r AS SELECT * FROM read_parquet({s(part)})')
        rows=con.execute('SELECT count(*),count(DISTINCT review_uid) FROM r').fetchone()
        assert rows[0]==rows[1]==pq.ParquetFile(part).metadata.num_rows
        predicates=[f'year IS NULL OR year<>{a.year}',
                    'gmap_id IS NULL OR normalized_text_hash IS NULL OR common_eligible_substantive IS NULL',
                    'normalized_duplicate_count_within_place IS NULL OR normalized_duplicate_count_within_place<1',
                    'normalized_duplicate_count_within_place<>floor(normalized_duplicate_count_within_place)',
                    'within_place_duplicate_weight IS NULL OR NOT isfinite(within_place_duplicate_weight)',
                    'abs(within_place_duplicate_weight-1.0/normalized_duplicate_count_within_place)>1e-12']
        for col in raw29+[TARGET[t]+'__evidence_probability' for t in RAW]:
            predicates.append(f'{q(col)} IS NULL OR NOT isfinite(CAST({q(col)} AS DOUBLE))')
        for tid in RAW:
            p=q(TARGET[tid]+'__evidence_probability'); c=q(TARGET[tid]+'__fixed_p05_call')
            predicates.append(f'{c} IS NULL OR CAST({c} AS BOOLEAN) IS DISTINCT FROM ({p}>=.5)')
        assert con.execute('SELECT count(*) FROM r WHERE '+' OR '.join('('+x+')' for x in predicates)).fetchone()[0]==0
        con.execute('CREATE OR REPLACE TABLE c AS '+cluster_sql())
        # Every complete hash cluster MUST reside in this bucket. A split cluster
        # fails raw_rows==declared_count instead of being double-counted as units.
        assert con.execute('SELECT count(*) FROM c WHERE declared_min<>declared_max OR raw_rows<>declared_max OR abs(all_mass-1)>1e-9').fetchone()[0]==0
        conflicts=con.execute('SELECT count(*) FROM c WHERE eligibility_min<>eligibility_max').fetchone()[0]
        cols=['gmap_id','sum(raw_rows)::BIGINT AS raw_rows','sum(eligible_rows)::BIGINT AS eligible_rows',
              'count(*)::BIGINT AS all_clusters','count(*) FILTER(WHERE eligible_mass>0)::BIGINT AS eligible_clusters',
              'sum(all_mass) AS all_mass','sum(eligible_mass) AS eligible_mass',
              'sum(all_mass*all_mass) AS all_mass_squared','sum(eligible_mass*eligible_mass) AS eligible_mass_squared']
        cols += [f'sum({q(col)}) AS {q(col)}' for col in sums]
        partial=partial_dir/(part.stem+'.place_partial.parquet')
        con.execute(f'COPY (SELECT '+','.join(cols)+f' FROM c GROUP BY gmap_id) TO {s(partial)} (FORMAT PARQUET,COMPRESSION ZSTD)')
        inventory.append(dict(path=str(part),sha256=digest,rows=rows[0],receipt_sha256=sha(receipt_path),
                              partial_sha256=sha(partial),eligibility_conflict_clusters=conflicts))
        total_rows+=rows[0]; conflict_clusters+=conflicts
        print(json.dumps(dict(parts_complete=len(inventory),parts_total=len(parts),rows=total_rows)),flush=True)
    assert total_rows==a.expected_rows
    assert seen_inputs==set(substrate)
    paths='['+','.join(s(p) for p in sorted(partial_dir.glob('*.parquet')))+']'
    additive=['raw_rows','eligible_rows','all_clusters','eligible_clusters','all_mass','eligible_mass',
              'all_mass_squared','eligible_mass_squared']+sums
    con.execute('CREATE OR REPLACE TABLE places AS SELECT gmap_id,'+','.join(f'sum({q(c)}) AS {q(c)}' for c in additive)+f' FROM read_parquet({paths}) GROUP BY gmap_id')
    # Global UID uniqueness is checked by the frozen finalized-substrate audit;
    # bucket completeness above additionally checks aggregation's exact units.
    assert con.execute('SELECT sum(raw_rows) FROM places').fetchone()[0]==total_rows
    select=['gmap_id',f'{a.year}::INTEGER AS review_year',
            'raw_rows::BIGINT AS raw_review_count','eligible_rows::BIGINT AS eligible_review_count',
            'all_clusters::BIGINT AS L0__clusters','eligible_clusters::BIGINT AS L1__clusters',
            'all_mass AS L0__effective_cluster_mass','eligible_mass AS eligible_effective_cluster_mass',
            'all_mass*all_mass/all_mass_squared AS L0__effective_sample_size',
            'eligible_mass*eligible_mass/nullif(eligible_mass_squared,0) AS eligible_effective_sample_size',
            '(eligible_clusters>=50) AS eligible_clusters_ge50']
    for tid in RAW:
        for level in ['L0','L1','L2','L3']:
            denom='all_mass' if level=='L0' else 'nullif(eligible_mass,0)'
            select.append(f'{q(level+"_"+tid+"_sum")}/{denom} AS {q(level+"__"+tid)}')
    out=a.output_dir/f'PLACE_LEVELS_{a.year}_FOUR_TARGETS.parquet'
    con.execute('COPY (SELECT '+','.join(select)+f' FROM places) TO {s(out)} (FORMAT PARQUET,COMPRESSION ZSTD)')
    place_n=con.execute('SELECT count(*) FROM places').fetchone()[0]
    con.close()
    write_json(a.output_dir/'AGGREGATION_RECEIPT.json',dict(status='SCORES_COMPLETE_METADATA_ATTACHMENT_PENDING',
        year=a.year,rows=total_rows,places=place_n,input_parts=inventory,model_artifact_sha256=a.model_sha256,
        production_manifest_sha256=manifest_sha,sealed_enrichment_receipt_sha256=sha(a.sealed_enrichment_receipt),
        spec_sha256=sha(a.spec_file),code_sha256=sha(Path(__file__)),
        source_aggregation='frozen aggregate_places_v1_2_8.py; annual scalar means only, no CI output',
        arithmetic='sum row 1/global2018_duplicate_n times value; eligible numerator and denominator retain mixed clusters',
        L2_interpretation='uncalibrated_not_prevalence',cutoff=.5,eligibility_conflict_clusters=conflict_clusters,
        metadata_complete=False,new_models=0,retuning=False,output_path=str(out),output_sha256=sha(out),
        elapsed_seconds=time.time()-started))


if __name__=='__main__': main()
