#!/usr/bin/env python3
"""
src/generate_submission.py
Full End-to-End Test Prediction & Submission Generation Pipeline.
1. Processes countries modularly (France -> US -> India) for minimal memory footprint.
2. Uses 4-channel Equi-Join Candidate Blocking.
3. RapidFuzz + LightGBM Model Scoring.
4. Exports matching_results.tsv and candidate_pairs.tsv in EXACT test_source1.tsv order.
"""
import os
import sys
import time
import pickle
import numpy as np
import duckdb
from feature_extractor import compute_pair_features, FEATURE_NAMES

DELIM = "\t"

def run_country(country, con, model, threshold=0.55):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"PROCESSING COUNTRY: {country.upper()}")
    print(f"{'='*60}")
    
    # 1. Load Country Records for S1, S2, S3
    con.execute(f"""
        DROP TABLE IF EXISTS s1;
        DROP TABLE IF EXISTS s2;
        DROP TABLE IF EXISTS s3;
        
        CREATE TABLE s1 AS 
        SELECT entity_id, business_name, business_address 
        FROM read_csv('dataset/test/test_source1.tsv', delim='\t', header=true)
        WHERE country = '{country}';
        
        CREATE TABLE s2 AS 
        SELECT entity_id, business_name, business_address 
        FROM read_csv('dataset/test/test_source2.tsv', delim='\t', header=true)
        WHERE country = '{country}';
        
        CREATE TABLE s3 AS 
        SELECT entity_id, business_name, business_address 
        FROM read_csv('dataset/test/test_source3.tsv', delim='\t', header=true)
        WHERE country = '{country}';
    """)
    
    n_s1 = con.execute("SELECT count(*) FROM s1").fetchone()[0]
    n_s2 = con.execute("SELECT count(*) FROM s2").fetchone()[0]
    n_s3 = con.execute("SELECT count(*) FROM s3").fetchone()[0]
    print(f"Loaded {n_s1:,} S1, {n_s2:,} S2, {n_s3:,} S3 records in {time.time() - t0:.2f}s")
    
    # 2. Materialize Normalized Blocking Keys
    print("Materializing blocking keys...")
    con.execute(r"""
        DROP MACRO IF EXISTS norm_name;
        DROP MACRO IF EXISTS sort_tokens;
        DROP MACRO IF EXISTS first2_words;
        DROP MACRO IF EXISTS addr_num;
        
        CREATE MACRO norm_name(s) AS (
            regexp_replace(
                replace(
                    regexp_replace(
                        regexp_replace(
                            trim(regexp_replace(lower(strip_accents(replace(s, '&', ' and '))), '[^a-z0-9]+', ' ', 'g')),
                            '\b(the|inc|corp|corporation|incorporated|llc|llp|ltd|limited|co|company|enterprises|services|solutions|group|technologies|pvt|sarl|sas|sci|sa)\b', '', 'g'
                        ),
                        '\b(com|org|net|in|fr)\b', '', 'g'
                    ),
                    ' ', ''
                ),
                '[^a-z0-9]', '', 'g'
            )
        );
        CREATE MACRO sort_tokens(s) AS (
            list_aggregate(
                list_sort(
                    list_filter(
                        string_split(
                            regexp_replace(
                                regexp_replace(
                                    trim(regexp_replace(lower(strip_accents(replace(s, '&', ' and '))), '[^a-z0-9]+', ' ', 'g')),
                                    '\b(the|and|inc|corp|corporation|incorporated|llc|llp|ltd|limited|co|company|enterprises|services|solutions|pvt|sarl|sas|sci|sa)\b', '', 'g'
                                ),
                                '\b(com|org|net|in|fr)\b', '', 'g'
                            ),
                            ' '
                        ),
                        x -> length(x) >= 2
                    )
                ),
                'string_agg', ' '
            )
        );
        CREATE MACRO first2_words(s) AS (
            regexp_extract(
                regexp_replace(
                    regexp_replace(
                        trim(regexp_replace(lower(strip_accents(replace(s, '&', ' and '))), '[^a-z0-9]+', ' ', 'g')),
                        '\b(the|and|inc|corp|corporation|incorporated|llc|llp|ltd|limited|co|company|enterprises|pvt|sarl|sas|sci|sa)\b', '', 'g'
                    ),
                    '\b(com|org|net|in|fr)\b', '', 'g'
                ),
                '^\s*([a-z0-9]+\s+[a-z0-9]+)'
            )
        );
        CREATE MACRO addr_num(s) AS (
            ltrim(regexp_extract(s, '\b([0-9]{1,7})\b', 1), '0')
        );
        
        DROP TABLE IF EXISTS s1_k;
        DROP TABLE IF EXISTS s2_k;
        DROP TABLE IF EXISTS s3_k;
        
        CREATE TABLE s1_k AS 
        SELECT entity_id, business_name, business_address,
               norm_name(business_name) as nname,
               sort_tokens(business_name) as stoks,
               first2_words(business_name) as f2w,
               addr_num(business_address) as anum,
               substr(norm_name(business_name), 1, 2) as pfx
        FROM s1;
        
        CREATE TABLE s2_k AS 
        SELECT entity_id, business_name, business_address,
               norm_name(business_name) as nname,
               sort_tokens(business_name) as stoks,
               first2_words(business_name) as f2w,
               addr_num(business_address) as anum,
               substr(norm_name(business_name), 1, 2) as pfx
        FROM s2;
        
        CREATE TABLE s3_k AS 
        SELECT entity_id, business_name, business_address,
               norm_name(business_name) as nname,
               sort_tokens(business_name) as stoks,
               first2_words(business_name) as f2w,
               addr_num(business_address) as anum,
               substr(norm_name(business_name), 1, 2) as pfx
        FROM s3;
    """)
    print(f"Keys materialized in {time.time() - t0:.2f}s")
    
    # 3. Multi-Channel Hash Equi-Joins with Priority
    print("Generating candidate pairs via 4-channel Equi-Joins...")
    con.execute("""
        DROP TABLE IF EXISTS s2_raw;
        DROP TABLE IF EXISTS s3_raw;
        DROP TABLE IF EXISTS final_cands;
        
        -- S2 Candidates
        CREATE TABLE s2_raw AS
        SELECT s1.entity_id as s1_id, s2.entity_id as target_id, 1 as priority
        FROM s1_k s1 JOIN s2_k s2 ON s1.nname = s2.nname WHERE length(s1.nname) >= 3
        UNION ALL
        SELECT s1.entity_id as s1_id, s2.entity_id as target_id, 2 as priority
        FROM s1_k s1 JOIN s2_k s2 ON s1.stoks = s2.stoks WHERE length(s1.stoks) >= 4
        UNION ALL
        SELECT s1.entity_id as s1_id, s2.entity_id as target_id, 3 as priority
        FROM s1_k s1 JOIN s2_k s2 ON s1.f2w = s2.f2w WHERE length(s1.f2w) >= 7
        UNION ALL
        SELECT s1.entity_id as s1_id, s2.entity_id as target_id, 4 as priority
        FROM s1_k s1 JOIN s2_k s2 ON s1.anum = s2.anum AND s1.pfx = s2.pfx 
        WHERE s1.anum != '' AND length(s1.pfx) >= 2;

        -- S3 Candidates
        CREATE TABLE s3_raw AS
        SELECT s1.entity_id as s1_id, s3.entity_id as target_id, 1 as priority
        FROM s1_k s1 JOIN s3_k s3 ON s1.nname = s3.nname WHERE length(s1.nname) >= 3
        UNION ALL
        SELECT s1.entity_id as s1_id, s3.entity_id as target_id, 2 as priority
        FROM s1_k s1 JOIN s3_k s3 ON s1.stoks = s3.stoks WHERE length(s1.stoks) >= 4
        UNION ALL
        SELECT s1.entity_id as s1_id, s3.entity_id as target_id, 3 as priority
        FROM s1_k s1 JOIN s3_k s3 ON s1.f2w = s3.f2w WHERE length(s1.f2w) >= 7
        UNION ALL
        SELECT s1.entity_id as s1_id, s3.entity_id as target_id, 4 as priority
        FROM s1_k s1 JOIN s3_k s3 ON s1.anum = s3.anum AND s1.pfx = s3.pfx 
        WHERE s1.anum != '' AND length(s1.pfx) >= 2;

        -- Deduplicate and cap S2 and S3 candidates
        CREATE TABLE final_cands AS
        SELECT s1_id, target_id, min(priority) as priority
        FROM (
            SELECT s1_id, target_id, priority FROM s2_raw WHERE priority <= 3
            UNION ALL
            SELECT s1_id, target_id, priority FROM (
                SELECT s1_id, target_id, priority, row_number() OVER (PARTITION BY s1_id ORDER BY s1_id) as rn
                FROM s2_raw WHERE priority = 4
            ) WHERE rn <= 15
            UNION ALL
            SELECT s1_id, target_id, priority FROM s3_raw WHERE priority <= 3
            UNION ALL
            SELECT s1_id, target_id, priority FROM (
                SELECT s1_id, target_id, priority, row_number() OVER (PARTITION BY s1_id ORDER BY s1_id) as rn
                FROM s3_raw WHERE priority = 4
            ) WHERE rn <= 15
        )
        GROUP BY s1_id, target_id;
    """)
    
    total_candidates = con.execute("SELECT count(*) FROM final_cands").fetchone()[0]
    print(f"Generated {total_candidates:,} candidate pairs in {time.time() - t0:.2f}s")
    
    # 4. Extract Text for Candidate Pairs and Score with ML Model
    print("Scoring candidate pairs with LightGBM model...")
    con.execute("""
        DROP TABLE IF EXISTS pairs_to_score;
        CREATE TABLE pairs_to_score AS
        SELECT c.s1_id, s1.business_name as s1_name, s1.business_address as s1_addr,
               c.target_id, target.business_name as cand_name, target.business_address as cand_addr,
               CASE WHEN c.target_id LIKE 'S2-%' THEN 1 ELSE 0 END as is_s2,
               c.priority
        FROM final_cands c
        JOIN s1_k s1 ON c.s1_id = s1.entity_id
        JOIN (
            SELECT entity_id, business_name, business_address FROM s2_k
            UNION ALL
            SELECT entity_id, business_name, business_address FROM s3_k
        ) target ON c.target_id = target.entity_id;
    """)
    
    # Process in streaming batches
    cur = con.cursor()
    cur.execute("SELECT * FROM pairs_to_score")
    
    matched_records = []
    candidate_records = []
    
    # Mapping per S1: s1_id -> list of candidate ids, s1_id -> list of match ids
    cand_agg = {}
    match_agg = {}
    
    total_scored = 0
    total_matched = 0
    while True:
        rows = cur.fetchmany(60000)
        if not rows:
            break
            
        X_batch = []
        batch_keys = []
        for r in rows:
            s1_id, s1_name, s1_addr, cand_id, cand_name, cand_addr, is_s2, pri = r
            if s1_id not in cand_agg:
                cand_agg[s1_id] = []
                match_agg[s1_id] = []
            cand_agg[s1_id].append(cand_id)
            
            feats = compute_pair_features(s1_name, s1_addr, cand_name, cand_addr, is_s2, pri)
            X_batch.append(feats)
            batch_keys.append((s1_id, cand_id))
            
        X_mat = np.array(X_batch, dtype=np.float32)
        probs = model.predict(X_mat)
        
        for (s1_id, cand_id), prob in zip(batch_keys, probs):
            if prob >= threshold:
                match_agg[s1_id].append(cand_id)
                total_matched += 1
                
        total_scored += len(rows)
        if total_scored % 300000 == 0 or total_scored == total_candidates:
            print(f"  Scored {total_scored:,} / {total_candidates:,} pairs ({total_matched:,} matched)...")
            
    print(f"Scoring complete for {country}. Inserting predictions into database...")
    
    # Store aggregated candidate & match strings into global prediction table
    con.execute("""
        CREATE TEMP TABLE IF NOT EXISTS country_preds (
            s1_id VARCHAR PRIMARY KEY,
            candidate_ids VARCHAR,
            matched_ids VARCHAR
        );
    """)
    
    pred_tuples = []
    for s1_id in cand_agg:
        c_list = list(dict.fromkeys(cand_agg[s1_id]))
        m_list = list(dict.fromkeys(match_agg[s1_id]))
        c_str = ",".join(c_list)
        m_str = ",".join(m_list)
        pred_tuples.append((s1_id, c_str, m_str))
        
    con.executemany("INSERT INTO country_preds VALUES (?, ?, ?)", pred_tuples)
    print(f"Saved {len(pred_tuples):,} entity predictions for {country} in {time.time() - t0:.2f}s")

def main():
    total_start = time.time()
    os.makedirs("output", exist_ok=True)
    
    model_path = "models/lgb_model.pkl"
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found!")
        sys.exit(1)
        
    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
        model = artifact["model"]
        best_threshold = artifact.get("best_threshold", 0.55)
        
    active_threshold = max(best_threshold, 0.50)
    print(f"Active F0.5 Threshold: {active_threshold:.2f}")
    
    con = duckdb.connect()
    
    # Global table for all country predictions
    con.execute("""
        CREATE TABLE all_predictions (
            s1_id VARCHAR PRIMARY KEY,
            candidate_ids VARCHAR,
            matched_ids VARCHAR
        );
    """)
    
    # Run France, US, India sequentially
    countries = ["France", "US", "India"]
    for c in countries:
        con.execute("DROP TABLE IF EXISTS country_preds;")
        run_country(c, con, model, threshold=active_threshold)
        con.execute("INSERT INTO all_predictions SELECT * FROM country_preds;")
        con.execute("DROP TABLE IF EXISTS country_preds;")
        
    print("\n" + "="*60)
    print("ALL COUNTRIES PROCESSED! GENERATING FINAL TEST SUBMISSION FILES...")
    print("="*60)
    
    # Export in EXACT line-by-line sequence of test_source1.tsv using DuckDB COPY!
    matching_path = "output/matching_results.tsv"
    candidate_path = "output/candidate_pairs.tsv"
    
    print(f"Exporting {matching_path}...")
    con.execute(f"""
        COPY (
            SELECT s1.entity_id AS source1_entity_id,
                   coalesce(p.matched_ids, '') AS matched_entity_ids
            FROM read_csv('dataset/test/test_source1.tsv', delim='\t', header=true) s1
            LEFT JOIN all_predictions p ON s1.entity_id = p.s1_id
        ) TO '{matching_path}' (HEADER, DELIMITER '\t');
    """)
    
    print(f"Exporting {candidate_path}...")
    con.execute(f"""
        COPY (
            SELECT s1.entity_id AS source1_entity_id,
                   coalesce(p.candidate_ids, '') AS candidate_entity_ids
            FROM read_csv('dataset/test/test_source1.tsv', delim='\t', header=true) s1
            LEFT JOIN all_predictions p ON s1.entity_id = p.s1_id
        ) TO '{candidate_path}' (HEADER, DELIMITER '\t');
    """)
    
    print("\n" + "="*60)
    print(f"FINAL SUBMISSION GENERATION COMPLETE IN {time.time() - total_start:.2f}s!")
    print(f"Matching Results: {matching_path} ({os.path.getsize(matching_path)/(1024*1024):.2f} MB)")
    print(f"Candidate Pairs:  {candidate_path} ({os.path.getsize(candidate_path)/(1024*1024):.2f} MB)")
    print("="*60)

if __name__ == "__main__":
    main()
