#!/usr/bin/env python3
"""
src/generate_full_submission.py
Production script to generate official:
  1. output/matching_results.tsv
  2. output/candidate_pairs.tsv

Guarantees:
- ZERO double quotes (pure TSV per line 82 of README)
- Exactly 1,732,544 rows in exact test_source1.tsv sequence
- Matches are guaranteed 100% strict subset of candidates
- Singletons have empty field after tab (S1-xxx\t\n)
"""
import os
import sys
import time
import duckdb

DELIM = "\t"

def build_full_submission():
    total_start = time.time()
    os.makedirs("output", exist_ok=True)
    
    con = duckdb.connect()
    con.execute("PRAGMA threads=8;")
    con.execute("PRAGMA memory_limit='10GB';")
    
    # 1. Global tables for predictions
    con.execute("""
        CREATE TABLE all_matches (
            s1_id VARCHAR,
            target_id VARCHAR
        );
        CREATE TABLE all_extra_candidates (
            s1_id VARCHAR,
            target_id VARCHAR
        );
    """)
    
    countries = ["France", "US", "India"]
    
    for country in countries:
        t_c = time.time()
        print(f"\n{'='*60}")
        print(f"PROCESSING {country.upper()}...")
        print(f"{'='*60}")
        
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
        print(f"Loaded {n_s1:,} S1, {n_s2:,} S2, {n_s3:,} S3 records in {time.time() - t_c:.2f}s")
        
        con.execute(r"""
            DROP MACRO IF EXISTS norm_name;
            DROP MACRO IF EXISTS sort_tokens;
            DROP MACRO IF EXISTS addr_num;
            
            CREATE MACRO norm_name(s) AS (
                regexp_replace(
                    replace(
                        regexp_replace(
                            regexp_replace(
                                trim(regexp_replace(lower(strip_accents(replace(s, '&', ' and '))), '[^a-z0-9]+', ' ', 'g')),
                                '\b(the|inc|corp|corporation|incorporated|llc|llp|ltd|limited|co|company|enterprises|services|solutions|group|technologies|pvt|sarl|sas|sci|sa|eurl)\b', '', 'g'
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
                                        '\b(the|and|inc|corp|corporation|incorporated|llc|llp|ltd|limited|co|company|enterprises|services|solutions|pvt|sarl|sas|sci|sa|eurl)\b', '', 'g'
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
            CREATE MACRO addr_num(s) AS (
                ltrim(regexp_extract(s, '([0-9]+)', 1), '0')
            );
            
            DROP TABLE IF EXISTS s1_k;
            DROP TABLE IF EXISTS s2_k;
            DROP TABLE IF EXISTS s3_k;
            
            CREATE TABLE s1_k AS 
            SELECT entity_id, norm_name(business_name) as nname, sort_tokens(business_name) as stoks, addr_num(business_address) as anum FROM s1;
            
            CREATE TABLE s2_k AS 
            SELECT entity_id, norm_name(business_name) as nname, sort_tokens(business_name) as stoks, addr_num(business_address) as anum FROM s2;
            
            CREATE TABLE s3_k AS 
            SELECT entity_id, norm_name(business_name) as nname, sort_tokens(business_name) as stoks, addr_num(business_address) as anum FROM s3;
        """)
        
        print(f"Running Candidate & Match Generation for {country}...")
        con.execute("""
            DROP TABLE IF EXISTS raw_pairs;
            
            CREATE TABLE raw_pairs AS
            -- Priority 1: Exact Name + Address Number match
            SELECT s1.entity_id as s1_id, s2.entity_id as target_id, 1 as pri
            FROM s1_k s1 JOIN s2_k s2 ON s1.nname = s2.nname WHERE length(s1.nname) >= 3 AND (s1.anum != '' AND s1.anum = s2.anum)
            UNION ALL
            SELECT s1.entity_id as s1_id, s3.entity_id as target_id, 1 as pri
            FROM s1_k s1 JOIN s3_k s3 ON s1.nname = s3.nname WHERE length(s1.nname) >= 3 AND (s1.anum != '' AND s1.anum = s3.anum)
            
            -- Priority 2: Sorted Tokens + Address Number match
            UNION ALL
            SELECT s1.entity_id as s1_id, s2.entity_id as target_id, 2 as pri
            FROM s1_k s1 JOIN s2_k s2 ON s1.stoks = s2.stoks WHERE length(s1.stoks) >= 4 AND (s1.anum != '' AND s1.anum = s2.anum)
            UNION ALL
            SELECT s1.entity_id as s1_id, s3.entity_id as target_id, 2 as pri
            FROM s1_k s1 JOIN s3_k s3 ON s1.stoks = s3.stoks WHERE length(s1.stoks) >= 4 AND (s1.anum != '' AND s1.anum = s3.anum)
            
            -- Priority 3: Exact Normalized Name Match (For entities without street numbers or rare names)
            UNION ALL
            SELECT s1.entity_id as s1_id, s2.entity_id as target_id, 3 as pri
            FROM s1_k s1 JOIN s2_k s2 ON s1.nname = s2.nname WHERE length(s1.nname) >= 15 AND s1.anum = '' AND s2.anum = ''
            UNION ALL
            SELECT s1.entity_id as s1_id, s3.entity_id as target_id, 3 as pri
            FROM s1_k s1 JOIN s3_k s3 ON s1.nname = s3.nname WHERE length(s1.nname) >= 15 AND s1.anum = '' AND s3.anum = ''
            
            -- Priority 4: Candidate-only pool (sorted tokens length >= 5)
            UNION ALL
            SELECT s1.entity_id as s1_id, s2.entity_id as target_id, 4 as pri
            FROM s1_k s1 JOIN s2_k s2 ON s1.stoks = s2.stoks WHERE length(s1.stoks) >= 5
            UNION ALL
            SELECT s1.entity_id as s1_id, s3.entity_id as target_id, 4 as pri
            FROM s1_k s1 JOIN s3_k s3 ON s1.stoks = s3.stoks WHERE length(s1.stoks) >= 5;
        """)
        
        # Deduplicate and Cap Candidates & Matches
        con.execute("""
            DROP TABLE IF EXISTS ranked_pairs;
            CREATE TABLE ranked_pairs AS
            SELECT s1_id, target_id, min(pri) as min_p,
                   row_number() OVER (PARTITION BY s1_id ORDER BY min(pri) ASC) as rn
            FROM raw_pairs
            GROUP BY s1_id, target_id;
            
            -- Final Matches: High-confidence top 5 (Priority 1, 2, or 3)
            INSERT INTO all_matches
            SELECT s1_id, target_id FROM ranked_pairs WHERE rn <= 5 AND min_p <= 3;
            
            -- Extra candidates: Remaining candidates up to 15
            INSERT INTO all_extra_candidates
            SELECT s1_id, target_id FROM ranked_pairs WHERE rn > 5 AND rn <= 15;
        """)
        
        m_cnt = con.execute("SELECT count(*) FROM all_matches").fetchone()[0]
        print(f"{country.upper()} done in {time.time() - t_c:.2f}s! Cumulative Matches: {m_cnt:,}")
        
    print("\n" + "="*60)
    print("ALL COUNTRIES PROCESSED. BUILDING LOOKUP MAPS...")
    print("="*60)
    
    # 2. Build In-Memory Match and Candidate Maps
    t_map = time.time()
    matches_raw = con.execute("SELECT s1_id, target_id FROM all_matches").fetchall()
    extra_cands_raw = con.execute("SELECT s1_id, target_id FROM all_extra_candidates").fetchall()
    
    match_dict = {}
    for s1_id, t_id in matches_raw:
        if s1_id not in match_dict:
            match_dict[s1_id] = []
        match_dict[s1_id].append(t_id)
        
    cand_dict = {}
    for s1_id, m_list in match_dict.items():
        cand_dict[s1_id] = list(m_list) # Guaranteed superset!
        
    for s1_id, t_id in extra_cands_raw:
        if s1_id not in cand_dict:
            cand_dict[s1_id] = []
        cand_dict[s1_id].append(t_id)
        
    print(f"Dictionaries built in {time.time() - t_map:.2f}s")
    
    # 3. Stream test_source1.tsv in EXACT order and write clean TSV without quotes
    matching_path = "output/matching_results.tsv"
    candidate_path = "output/candidate_pairs.tsv"
    
    print(f"Streaming and writing {matching_path} and {candidate_path} in exact test_source1 order...")
    t_write = time.time()
    
    total_written = 0
    empty_matches = 0
    
    with open("dataset/test/test_source1.tsv", "r", encoding="utf-8") as in_f, \
         open(matching_path, "w", encoding="utf-8") as m_f, \
         open(candidate_path, "w", encoding="utf-8") as c_f:
             
        # Write headers
        m_f.write("source1_entity_id\tmatched_entity_ids\n")
        c_f.write("source1_entity_id\tcandidate_entity_ids\n")
        
        # Skip header of test_source1.tsv
        header = in_f.readline()
        
        for line in in_f:
            s1_id = line.split(DELIM, 1)[0].strip()
            if not s1_id:
                continue
            total_written += 1
            
            m_list = match_dict.get(s1_id, [])
            c_list = cand_dict.get(s1_id, [])
            
            # Deduplicate while preserving order
            m_clean = list(dict.fromkeys(m_list))
            c_clean = list(dict.fromkeys(c_list))
            
            m_str = ",".join(m_clean)
            c_str = ",".join(c_clean)
            
            if not m_str:
                empty_matches += 1
                
            # Write exact clean tab-separated lines with NO QUOTING
            m_f.write(f"{s1_id}\t{m_str}\n")
            c_f.write(f"{s1_id}\t{c_str}\n")
            
    print(f"Written {total_written:,} entities ({empty_matches:,} singletons) in {time.time() - t_write:.2f}s")
    print(f"Matching Results: {matching_path} ({os.path.getsize(matching_path)/(1024*1024):.2f} MB)")
    print(f"Candidate Pairs:  {candidate_path} ({os.path.getsize(candidate_path)/(1024*1024):.2f} MB)")
    print(f"Total time elapsed: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    build_full_submission()
