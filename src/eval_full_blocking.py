#!/usr/bin/env python3
"""
src/eval_full_blocking.py
Measure combined Recall and Candidate density across S2 and S3 using 4 Hash Equi-Join channels.
"""
import duckdb
import time

def run_evaluation(num_s1=20000, country="US"):
    t0 = time.time()
    con = duckdb.connect()
    
    print(f"=== FULL BLOCKING BENCHMARK: {num_s1:,} S1 entities ({country}) ===")
    
    # 1. Load S1 slice
    con.execute(f"""
        CREATE TABLE s1 AS 
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('dataset/train/train_source1.tsv', delim='\t', header=true)
        WHERE country = '{country}'
        LIMIT {num_s1};
    """)
    
    # 2. Load Ground Truth
    con.execute("""
        CREATE TABLE gt AS
        SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS target_id
        FROM read_csv('dataset/train/train_ground_truth.tsv', delim='\t', header=true)
        WHERE source1_entity_id IN (SELECT entity_id FROM s1) AND matched_entity_ids != '';
    """)
    
    total_gt = con.execute("SELECT count(*) FROM gt").fetchone()[0]
    s2_gt = con.execute("SELECT count(*) FROM gt WHERE target_id LIKE 'S2-%'").fetchone()[0]
    s3_gt = con.execute("SELECT count(*) FROM gt WHERE target_id LIKE 'S3-%'").fetchone()[0]
    print(f"Ground Truth Targets: Total={total_gt:,} (S2={s2_gt:,}, S3={s3_gt:,})")
    
    # 3. Load S2 and S3
    print("Loading S2 and S3...")
    con.execute(f"""
        CREATE TABLE s2 AS 
        SELECT entity_id, business_name, business_address 
        FROM read_csv('dataset/train/train_source2.tsv', delim='\t', header=true)
        WHERE country = '{country}';
        
        CREATE TABLE s3 AS 
        SELECT entity_id, business_name, business_address 
        FROM read_csv('dataset/train/train_source3.tsv', delim='\t', header=true)
        WHERE country = '{country}';
    """)
    print(f"Sources loaded in {time.time() - t0:.2f}s")
    
    # 4. Define text normalization macros
    con.execute(r"""
        CREATE MACRO norm_name(s) AS (
            regexp_replace(
                replace(
                    regexp_replace(
                        regexp_replace(
                            trim(regexp_replace(lower(replace(s, '&', ' and ')), '[^a-z0-9]+', ' ', 'g')),
                            '\b(the|inc|corp|corporation|incorporated|llc|llp|ltd|limited|co|company|enterprises|services|solutions|group|technologies|pvt)\b', '', 'g'
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
                                    trim(regexp_replace(lower(replace(s, '&', ' and ')), '[^a-z0-9]+', ' ', 'g')),
                                    '\b(the|and|inc|corp|corporation|incorporated|llc|llp|ltd|limited|co|company|enterprises|services|solutions|pvt)\b', '', 'g'
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
                        trim(regexp_replace(lower(replace(s, '&', ' and ')), '[^a-z0-9]+', ' ', 'g')),
                        '\b(the|and|inc|corp|corporation|incorporated|llc|llp|ltd|limited|co|company|enterprises|pvt)\b', '', 'g'
                    ),
                    '\b(com|org|net)\b', '', 'g'
                ),
                '^\s*([a-z0-9]+\s+[a-z0-9]+)'
            )
        );
        CREATE MACRO addr_num(s) AS (
            ltrim(regexp_extract(s, '\b([0-9]{1,7})\b', 1), '0')
        );
    """)
    
    # 5. Materialize Keys for S1, S2, S3
    print("Materializing blocking keys...")
    con.execute("""
        CREATE TABLE s1_k AS 
        SELECT entity_id, 
               norm_name(business_name) as nname,
               sort_tokens(business_name) as stoks,
               first2_words(business_name) as f2w,
               addr_num(business_address) as anum,
               substr(norm_name(business_name), 1, 2) as pfx
        FROM s1;
        
        CREATE TABLE s2_k AS 
        SELECT entity_id, 
               norm_name(business_name) as nname,
               sort_tokens(business_name) as stoks,
               first2_words(business_name) as f2w,
               addr_num(business_address) as anum,
               substr(norm_name(business_name), 1, 2) as pfx
        FROM s2;
        
        CREATE TABLE s3_k AS 
        SELECT entity_id, 
               norm_name(business_name) as nname,
               sort_tokens(business_name) as stoks,
               first2_words(business_name) as f2w,
               addr_num(business_address) as anum,
               substr(norm_name(business_name), 1, 2) as pfx
        FROM s3;
    """)
    print(f"Keys materialized in {time.time() - t0:.2f}s")
    
    # 6. Generate Candidates via Union of Equi-Joins with Priority Tiers
    print("Running multi-channel equi-joins with priority ranking...")
    con.execute("""
        -- S2 matches
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

        -- S3 matches
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

        -- Combine: Keep ALL Priority 1, 2, 3 (exact/sorted/first2) and cap Priority 4 to top 20 per S1
        CREATE TABLE cands AS
        -- S2 High Priority
        SELECT s1_id, target_id, priority FROM s2_raw WHERE priority <= 3
        UNION ALL
        -- S2 Priority 4 capped to top 20
        SELECT s1_id, target_id, priority
        FROM (
            SELECT s1_id, target_id, priority,
                   row_number() OVER (PARTITION BY s1_id ORDER BY s1_id) as rn
            FROM s2_raw WHERE priority = 4
        ) WHERE rn <= 20
        UNION ALL
        -- S3 High Priority
        SELECT s1_id, target_id, priority FROM s3_raw WHERE priority <= 3
        UNION ALL
        -- S3 Priority 4 capped to top 20
        SELECT s1_id, target_id, priority
        FROM (
            SELECT s1_id, target_id, priority,
                   row_number() OVER (PARTITION BY s1_id ORDER BY s1_id) as rn
            FROM s3_raw WHERE priority = 4
        ) WHERE rn <= 20;
    """)
    
    # 7. Evaluate Recall against Ground Truth
    matched_s2 = con.execute("""
        SELECT count(DISTINCT gt.target_id)
        FROM gt JOIN cands c ON gt.source1_entity_id = c.s1_id AND gt.target_id = c.target_id
        WHERE gt.target_id LIKE 'S2-%';
    """).fetchone()[0]
    
    matched_s3 = con.execute("""
        SELECT count(DISTINCT gt.target_id)
        FROM gt JOIN cands c ON gt.source1_entity_id = c.s1_id AND gt.target_id = c.target_id
        WHERE gt.target_id LIKE 'S3-%';
    """).fetchone()[0]
    
    total_matched = matched_s2 + matched_s3
    
    stats = con.execute(f"""
        SELECT 
            count(*) as total_cands,
            count(DISTINCT s1_id) as s1_with_cands,
            count(*) * 1.0 / {num_s1} as avg_cands,
            quantile_cont(cnt, 0.5) as median_cands,
            quantile_cont(cnt, 0.95) as p95_cands,
            max(cnt) as max_cands
        FROM (SELECT s1_id, count(*) as cnt FROM cands GROUP BY s1_id);
    """).fetchone()
    
    print("\n" + "="*60)
    print("--- FULL BLOCKING PIPELINE RESULTS ---")
    print(f"Target S1 Entities:            {num_s1:,}")
    print(f"S2 True Matches Captured:      {matched_s2:,} / {s2_gt:,} ({matched_s2 / s2_gt * 100:.2f}%)")
    print(f"S3 True Matches Captured:      {matched_s3:,} / {s3_gt:,} ({matched_s3 / s3_gt * 100:.2f}%)")
    print(f"OVERALL BLOCKING RECALL:       {total_matched:,} / {total_gt:,} ({total_matched / total_gt * 100:.2f}%)")
    print("-" * 60)
    print(f"Total Candidates Generated:    {stats[0]:,}")
    print(f"S1 Entities with Candidates:   {stats[1]:,} / {num_s1:,} ({stats[1]/num_s1*100:.2f}%)")
    print(f"Average Candidates per S1:     {stats[2]:.2f}")
    print(f"Median Candidates per S1:      {stats[3]:.0f}")
    print(f"95th Percentile Candidates:    {stats[4]:.0f}")
    print(f"Max Candidates for single S1:  {stats[5]}")
    print(f"TOTAL EXECUTION TIME:          {time.time() - t0:.2f}s")
    print("="*60)

if __name__ == "__main__":
    run_evaluation(num_s1=20000, country="US")
