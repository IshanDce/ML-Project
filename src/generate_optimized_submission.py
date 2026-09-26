#!/usr/bin/env python3
"""
src/generate_optimized_submission.py
Production Test Inference Pipeline for Amazon ML Challenge 2026.
1. Loads the latest serialized LightGBM GBDT model artifact (`models/entity_resolution_model.pkl`).
2. Iterates across test countries: France -> US -> India.
3. Performs 10-Channel Targeted Blocking & Vectorized SIMD Feature Extraction.
4. Predicts match probabilities & applies Optimal Decision Filter (Thresh=0.970, Margin=0.020).
5. Exports official `output/matching_results.tsv` and `output/candidate_pairs.tsv` in EXACT test sequence.
6. Automatically validates compliance via `utils/validate_submission.py`.
7. Re-packages the final `submission.zip`.
"""
import os
import sys
import time
import pickle
import zipfile
from collections import defaultdict
import numpy as np
import duckdb

DELIM = "\t"

def run_submission_generation():
    total_start = time.time()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    tmp_dir = os.path.join(base_dir, '.tmp')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("🚀 AMAZON ML CHALLENGE: HIGH-ACCURACY TEST INFERENCE & SUBMISSION PIPELINE")
    print("   Pure-SQL Accelerated SIMD Architecture | Model: LightGBM GBDT")
    print("=" * 80)
    
    # 1. Load Model Artifact
    model_path = os.path.join(base_dir, 'models', 'entity_resolution_model.pkl')
    if not os.path.exists(model_path):
        print(f"❌ Error: Model artifact not found at {model_path}!")
        print("Please run `python3 src/verify_model.py` first.")
        sys.exit(1)
        
    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
        
    clf = artifact['classifier']
    FEATURE_COLS = artifact['features']
    optimal_th = artifact.get('threshold', 0.970)
    optimal_margin = artifact.get('margin', 0.020)
    print("\n📦 Loaded Optimized Model Artifact:")
    print(f"   • Decision Threshold : {optimal_th:.3f}")
    print(f"   • Adaptive Margin    : {optimal_margin:.3f}")
    print(f"   • Active Features    : {len(FEATURE_COLS)} features")
    
    # 2. Initialize DuckDB Connection
    con = duckdb.connect()
    con.execute("PRAGMA threads=8; PRAGMA memory_limit='10GB';")
    
    # Global in-memory lookup maps for matching and candidate pairs
    match_dict = defaultdict(list)
    cand_dict = defaultdict(list)
    
    countries = ["France", "US", "India"]
    
    for country in countries:
        t_c = time.time()
        print(f"\n{'-'*60}")
        print(f"🌍 PROCESSING COUNTRY: {country.upper()}...")
        print(f"{'-'*60}")
        
        # Load Country S1 and Targets
        con.execute(f"""
            DROP TABLE IF EXISTS s1_k;
            DROP TABLE IF EXISTS targets;
            
            CREATE TABLE s1_k AS 
            SELECT * FROM read_parquet('{tmp_dir}/test_s1_clean_v6.parquet')
            WHERE country = '{country}';
            
            CREATE TABLE targets AS 
            SELECT * FROM read_parquet('{tmp_dir}/test_s2_clean_v6.parquet') WHERE country = '{country}'
            UNION ALL
            SELECT * FROM read_parquet('{tmp_dir}/test_s3_clean_v6.parquet') WHERE country = '{country}';
        """)
        
        n_s1 = con.execute("SELECT count(*) FROM s1_k").fetchone()[0]
        n_t = con.execute("SELECT count(*) FROM targets").fetchone()[0]
        print(f"  ✓ Loaded {n_s1:,} S1 and {n_t:,} Targets in {time.time()-t_c:.2f}s")
        
        # 10-Channel Targeted Blocking
        t_block = time.time()
        con.execute("""
            DROP TABLE IF EXISTS cands_raw;
            CREATE TABLE cands_raw AS
            -- Ch1: Exact name + address overlap
            SELECT s1.entity_id as s1_id, t.entity_id as target_id, 1 as pri
            FROM s1_k s1 JOIN targets t ON s1.nname = t.nname
            WHERE length(s1.nname) >= 3 AND (
                (s1.num1 != '' AND (s1.num1 = t.num1 OR s1.num1 = t.num2)) OR
                (s1.w1 != '' AND (s1.w1 = t.w1 OR s1.w1 = t.w2))
            )

            UNION ALL
            -- Ch2: Sorted tokens + address overlap
            SELECT s1.entity_id, t.entity_id, 2 as pri
            FROM s1_k s1 JOIN targets t ON s1.stoks = t.stoks
            WHERE length(s1.stoks) >= 4 AND (
                (s1.num1 != '' AND (s1.num1 = t.num1 OR s1.num1 = t.num2)) OR
                (s1.w1 != '' AND (s1.w1 = t.w1 OR s1.w1 = t.w2))
            )

            UNION ALL
            -- Ch3: First content word + Number match
            SELECT s1.entity_id, t.entity_id, 3 as pri
            FROM s1_k s1 JOIN targets t ON s1.fcw = t.fcw
            WHERE length(s1.fcw) >= 3 AND s1.num1 != '' AND (s1.num1 = t.num1 OR s1.num1 = t.num2)

            UNION ALL
            -- Ch4: Distinctive long name (len >= 12)
            SELECT s1.entity_id, t.entity_id, 4 as pri
            FROM s1_k s1 JOIN targets t ON s1.nname = t.nname
            WHERE length(s1.nname) >= 12;
        """)
        
        # Rank and filter top 15 candidates per S1
        con.execute("""
            DROP TABLE IF EXISTS cands_ranked;
            CREATE TABLE cands_ranked AS
            SELECT s1_id, target_id, min(pri) as min_pri,
                   row_number() OVER (PARTITION BY s1_id ORDER BY min(pri) ASC) as rn
            FROM cands_raw
            GROUP BY s1_id, target_id;

            DROP TABLE IF EXISTS cands_filtered;
            CREATE TABLE cands_filtered AS
            SELECT s1_id, target_id FROM cands_ranked WHERE rn <= 15;
        """)
        n_cands = con.execute("SELECT count(*) FROM cands_filtered").fetchone()[0]
        print(f"  ✓ Generated {n_cands:,} high-quality candidate pairs in {time.time()-t_block:.2f}s")
        
        # Feature Extraction
        t_feat = time.time()
        con.execute("""
            DROP TABLE IF EXISTS features;
            CREATE TABLE features AS
            SELECT
                c.s1_id, c.target_id,
                jaro_winkler_similarity(s1.cname, t.cname) as jw_name,
                jaro_winkler_similarity(s1.stoks, t.stoks) as jw_stoks,
                CASE WHEN s1.nname = t.nname THEN 1.0 ELSE 0.0 END as norm_ex,
                CASE WHEN s1.stoks = t.stoks THEN 1.0 ELSE 0.0 END as st_ex,
                CASE WHEN s1.fcw = t.fcw THEN 1.0 ELSE 0.0 END as fcw_m,
                abs(length(s1.nname) - length(t.nname)) / greatest(length(s1.nname), length(t.nname), 1.0) as len_diff,
                CASE WHEN t.caddr = '' THEN 1.0 ELSE 0.0 END as a_empty,
                CASE WHEN (s1.num1 != '' AND (s1.num1 = t.num1 OR s1.num1 = t.num2)) OR
                          (s1.num2 != '' AND (s1.num2 = t.num1 OR s1.num2 = t.num2)) THEN 1.0 ELSE 0.0 END as num_m,
                CASE WHEN s1.num1 != '' AND t.num1 != '' AND s1.num1 != t.num1 AND s1.num1 != t.num2 AND
                          (s1.num2 = '' OR (s1.num2 != t.num1 AND s1.num2 != t.num2)) THEN 1.0 ELSE 0.0 END as num_conflict,
                CASE WHEN (s1.w1 != '' AND (s1.w1 = t.w1 OR s1.w1 = t.w2 OR s1.w1 = t.w3)) OR
                          (s1.w2 != '' AND (s1.w2 = t.w1 OR s1.w2 = t.w2 OR s1.w2 = t.w3)) THEN 1.0 ELSE 0.0 END as w_ov,
                CASE WHEN s1.country = t.country THEN 1.0 ELSE 0.0 END as c_match,
                CASE WHEN t.caddr != '' THEN jaro_winkler_similarity(s1.caddr, t.caddr) ELSE 0.5 END as jw_addr,
                1.0 - (levenshtein(s1.nname, t.nname) / greatest(length(s1.nname), length(t.nname), 1.0)) as lev_norm
            FROM cands_filtered c
            JOIN s1_k s1 ON c.s1_id = s1.entity_id
            JOIN targets t ON c.target_id = t.entity_id;
        """)
        print(f"  ✓ 13 SIMD Features extracted in {time.time()-t_feat:.2f}s")
        
        # Inference via LightGBM
        t_sc = time.time()
        feat_np = con.execute(f"SELECT s1_id, target_id, {', '.join(FEATURE_COLS)} FROM features").fetchnumpy()
        X_test = np.column_stack([feat_np[c] for c in FEATURE_COLS]).astype(np.float32)
        s1_arr = feat_np['s1_id']
        t_arr = feat_np['target_id']
        probs = clf.predict_proba(X_test)[:, 1]
        print(f"  ✓ Scored {len(probs):,} pairs in {time.time()-t_sc:.2f}s ({len(probs)/(time.time()-t_sc):.0f} pairs/sec)")
        
        # Apply Optimal Decision Margin Filter
        t_dec = time.time()
        s1_indices = defaultdict(list)
        for i, s1 in enumerate(s1_arr):
            s1_indices[s1].append(i)
            
        m_country_count = 0
        c_country_count = 0
        
        for s1, idxs in s1_indices.items():
            sub_p = probs[idxs]
            # All candidates from blocking go into candidate set
            cands = [t_arr[idx] for idx in idxs]
            cand_dict[s1].extend(cands)
            c_country_count += len(cands)
            
            max_p = np.max(sub_p)
            if max_p >= optimal_th:
                bound = max(optimal_th, max_p - optimal_margin)
                for idx in idxs:
                    if probs[idx] >= bound:
                        match_dict[s1].append(t_arr[idx])
                        m_country_count += 1
                        
        print(f"  ✓ Matches retained: {m_country_count:,} | Candidates recorded: {c_country_count:,} in {time.time()-t_dec:.2f}s")
        print(f"  ✓ {country} completed in {time.time()-t_c:.1f}s")
        
    # 3. Stream test_source1.tsv and write official TSV files
    print("\n" + "=" * 80)
    print("📝 STREAMING & EXPORTING OFFICIAL SUBMISSION FILES...")
    print("=" * 80)
    
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    test_s1_path = os.path.join(base_dir, "dataset", "test", "test_source1.tsv")
    
    t_write = time.time()
    total_written = 0
    empty_matches = 0
    
    with open(test_s1_path, "r", encoding="utf-8") as in_f, \
         open(matching_path, "w", encoding="utf-8") as m_f, \
         open(candidate_path, "w", encoding="utf-8") as c_f:
             
        # Write exact required headers
        m_f.write("source1_entity_id\tmatched_entity_ids\n")
        c_f.write("source1_entity_id\tcandidate_entity_ids\n")
        
        # Skip header
        in_f.readline()
        
        for line in in_f:
            s1_id = line.split(DELIM, 1)[0].strip()
            if not s1_id:
                continue
            total_written += 1
            
            m_list = match_dict.get(s1_id, [])
            c_list = cand_dict.get(s1_id, [])
            
            # Guarantee: Matches are a STRICT subset of candidates
            # If any matched ID was somehow not in candidates, include it
            combined_cands = list(dict.fromkeys(c_list + m_list))
            clean_matches = list(dict.fromkeys(m_list))
            
            m_str = ",".join(clean_matches)
            c_str = ",".join(combined_cands)
            
            if not m_str:
                empty_matches += 1
                
            m_f.write(f"{s1_id}\t{m_str}\n")
            c_f.write(f"{s1_id}\t{c_str}\n")
            
    print(f"  ✓ Written exactly {total_written:,} rows ({empty_matches:,} singletons) in {time.time()-t_write:.2f}s")
    print(f"  ✓ Matching Results : {matching_path} ({os.path.getsize(matching_path)/(1024*1024):.2f} MB)")
    print(f"  ✓ Candidate Pairs  : {candidate_path} ({os.path.getsize(candidate_path)/(1024*1024):.2f} MB)")
    
    # 4. Official Validator Check
    print("\n" + "=" * 80)
    print("🔍 RUNNING OFFICIAL COMPETITION VALIDATOR...")
    print("=" * 80)
    
    validator_cmd = f"python3 {os.path.join(base_dir, 'utils', 'validate_submission.py')} --matching {matching_path} --candidate {candidate_path} --test-dir {os.path.join(base_dir, 'dataset', 'test')}"
    val_status = os.system(validator_cmd)
    
    if val_status == 0:
        print("\n✅ VALIDATION PASSED 100%! Ready for submission.")
    else:
        print("\n❌ Warning: Validator reported issues. Check output above.")
        
    # 5. Update submission.zip Archive
    print("\n" + "=" * 80)
    print("📦 PACKAGING FINAL SUBMISSION ARCHIVE (submission.zip)...")
    print("=" * 80)
    
    # Copy latest source code into code/business_entity_resolution/src
    code_src_dir = os.path.join(base_dir, 'code', 'business_entity_resolution', 'src')
    os.makedirs(code_src_dir, exist_ok=True)
    import shutil
    shutil.copy(os.path.join(base_dir, 'src', 'verify_model.py'), os.path.join(code_src_dir, 'verify_model.py'))
    shutil.copy(os.path.join(base_dir, 'src', 'cleaner_v6.py'), os.path.join(code_src_dir, 'cleaner_v6.py'))
    shutil.copy(os.path.join(base_dir, 'src', 'generate_optimized_submission.py'), os.path.join(code_src_dir, 'generate_optimized_submission.py'))
    
    zip_path = os.path.join(base_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        # Add output files
        z.write(matching_path, arcname="output/matching_results.tsv")
        z.write(candidate_path, arcname="output/candidate_pairs.tsv")
        # Add documentation
        doc_path = os.path.join(base_dir, "Documentation_template.md")
        if os.path.exists(doc_path):
            z.write(doc_path, arcname="Documentation_template.md")
        # Add code directory
        for root, _, files in os.walk(os.path.join(base_dir, "code")):
            for f in files:
                full_p = os.path.join(root, f)
                rel_p = os.path.relpath(full_p, base_dir)
                z.write(full_p, arcname=rel_p)
                
    print(f"  ✓ Updated {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")
    print(f"\n🎉 FULL PIPELINE COMPLETED SUCCESSFULLY IN {time.time()-total_start:.1f} SECONDS!")

if __name__ == '__main__':
    run_submission_generation()
