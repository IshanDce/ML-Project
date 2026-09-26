#!/usr/bin/env python3
"""
src/eval_duckdb_blocking.py
Evaluate candidate generation coverage using DuckDB's vectorized query engine.
"""
import duckdb
import time

def evaluate():
    t0 = time.time()
    con = duckdb.connect()
    
    print("Testing DuckDB Candidate Blocking on 20,000 S1 records (US)...")
    
    # 1. Load sample S1 records
    con.execute("""
        CREATE TABLE s1 AS 
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('dataset/train/train_source1.tsv', delim='\t', header=true)
        WHERE country = 'US'
        LIMIT 20000;
    """)
    
    # 2. Load Ground Truth for these S1
    con.execute("""
        CREATE TABLE gt AS
        SELECT source1_entity_id, unnest(string_split(matched_entity_ids, ',')) AS target_id
        FROM read_csv('dataset/train/train_ground_truth.tsv', delim='\t', header=true)
        WHERE source1_entity_id IN (SELECT entity_id FROM s1) AND matched_entity_ids != '';
    """)
    
    total_gt = con.execute("SELECT count(*) FROM gt").fetchone()[0]
    s2_total_gt = con.execute("SELECT count(*) FROM gt WHERE target_id LIKE 'S2-%'").fetchone()[0]
    s3_total_gt = con.execute("SELECT count(*) FROM gt WHERE target_id LIKE 'S3-%'").fetchone()[0]
    print(f"Ground Truth targets: Total={total_gt:,} (S2={s2_total_gt:,}, S3={s3_total_gt:,})")
    
    # 3. Load S2 and S3 for US
    print("Loading S2 and S3 US...")
    con.execute("""
        CREATE TABLE s2 AS 
        SELECT entity_id, business_name, business_address 
        FROM read_csv('dataset/train/train_source2.tsv', delim='\t', header=true)
        WHERE country = 'US';
    """)
    con.execute("""
        CREATE TABLE s3 AS 
        SELECT entity_id, business_name, business_address 
        FROM read_csv('dataset/train/train_source3.tsv', delim='\t', header=true)
        WHERE country = 'US';
    """)
    print(f"Data loaded in {time.time() - t0:.2f}s")
    
    # 4. Define text normalization macro
    con.execute(r"""
        CREATE MACRO norm_name(s) AS (
            regexp_replace(
                replace(
                    regexp_replace(
                        regexp_replace(
                            trim(regexp_replace(lower(s), '[^a-z0-9]+', ' ', 'g')),
                            '\b(inc|corp|corporation|incorporated|llc|llp|ltd|limited|co|company|enterprises|services|solutions|group|technologies)\b', '', 'g'
                        ),
                        '\b(com|org|net)\b', '', 'g'
                    ),
                    ' ', ''
                ),
                '[^a-z0-9]', '', 'g'
            )
        );
    """)
    
    # 5. Measure coverage on S2
    s2_matches = con.execute("""
        SELECT count(DISTINCT gt.target_id)
        FROM gt
        JOIN s1 ON gt.source1_entity_id = s1.entity_id
        JOIN s2 ON gt.target_id = s2.entity_id
        WHERE norm_name(s1.business_name) = norm_name(s2.business_name)
          AND length(norm_name(s1.business_name)) >= 3;
    """).fetchone()[0]
    
    print(f"S2 exact/norm_name matches: {s2_matches:,} / {s2_total_gt:,} ({s2_matches/s2_total_gt*100:.2f}%)")
    
    # 6. Measure coverage on S3
    s3_matches = con.execute("""
        SELECT count(DISTINCT gt.target_id)
        FROM gt
        JOIN s1 ON gt.source1_entity_id = s1.entity_id
        JOIN s3 ON gt.target_id = s3.entity_id
        WHERE norm_name(s1.business_name) = norm_name(s3.business_name)
          AND length(norm_name(s1.business_name)) >= 3;
    """).fetchone()[0]
    
    print(f"S3 exact/norm_name matches: {s3_matches:,} / {s3_total_gt:,} ({s3_matches/s3_total_gt*100:.2f}%)")
    print(f"Combined coverage on single rule: {(s2_matches + s3_matches) / total_gt * 100:.2f}%")
    # Add Address Number extraction macro
    con.execute(r"""
        CREATE MACRO addr_num(s) AS (
            ltrim(regexp_extract(s, '\b([0-9]{1,7})\b', 1), '0')
        );
        CREATE MACRO norm_street(s) AS (
            regexp_replace(
                regexp_replace(
                    trim(regexp_replace(lower(s), '[^a-z0-9]+', ' ', 'g')),
                    '\b(st|street|rd|road|ave|avenue|dr|drive|blvd|boulevard|ct|court|ln|lane|hwy|highway|fl|floor|ste|suite|unit|apt|apartment)\b', '', 'g'
                ),
                '[^a-z0-9]', '', 'g'
            )
        );
    """)

    # Add sorted tokens macro to handle word transpositions and leading 'The'
    con.execute(r"""
        CREATE MACRO sort_tokens(s) AS (
            list_aggregate(
                list_sort(
                    list_filter(
                        string_split(
                            regexp_replace(
                                regexp_replace(
                                    trim(regexp_replace(lower(s), '[^a-z0-9]+', ' ', 'g')),
                                    '\b(the|and|inc|corp|corporation|incorporated|llc|llp|ltd|limited|co|company|pvt)\b', '', 'g'
                                ),
                                '\b(com|org|net)\b', '', 'g'
                            ),
                            ' '
                        ),
                        x -> length(x) >= 2
                    )
                ),
                'string_agg', ' '
            )
        );
    """)

    # Add first 2 words macro
    con.execute(r"""
        CREATE MACRO first2_words(s) AS (
            regexp_extract(
                regexp_replace(
                    regexp_replace(
                        trim(regexp_replace(lower(s), '[^a-z0-9]+', ' ', 'g')),
                        '\b(the|and|inc|corp|corporation|incorporated|llc|llp|ltd|limited|co|company|pvt)\b', '', 'g'
                    ),
                    '\b(com|org|net)\b', '', 'g'
                ),
                '^\s*([a-z0-9]+\s+[a-z0-9]+)'
            )
        );
    """)

    # Rule 1 + 2 + 3 + 4 on S2
    s2_combined4 = con.execute("""
        SELECT count(DISTINCT gt.target_id)
        FROM gt
        JOIN s1 ON gt.source1_entity_id = s1.entity_id
        JOIN s2 ON gt.target_id = s2.entity_id
        WHERE (norm_name(s1.business_name) = norm_name(s2.business_name) AND length(norm_name(s1.business_name)) >= 3)
           OR (sort_tokens(s1.business_name) = sort_tokens(s2.business_name) AND length(sort_tokens(s1.business_name)) >= 4)
           OR (first2_words(s1.business_name) != '' 
               AND first2_words(s1.business_name) = first2_words(s2.business_name)
               AND length(first2_words(s1.business_name)) >= 7)
           OR (addr_num(s1.business_address) != '' 
               AND addr_num(s1.business_address) = addr_num(s2.business_address)
               AND (substr(norm_name(s1.business_name), 1, 2) = substr(norm_name(s2.business_name), 1, 2)
                    OR substr(sort_tokens(s1.business_name), 1, 2) = substr(sort_tokens(s2.business_name), 1, 2)));
    """).fetchone()[0]

    print(f"\nS2 Combined Rule 1+2+3+4 matches: {s2_combined4:,} / {s2_total_gt:,} ({s2_combined4/s2_total_gt*100:.2f}%)")

    # Rule 1 + 2 + 3 + 4 on S3
    s3_combined4 = con.execute("""
        SELECT count(DISTINCT gt.target_id)
        FROM gt
        JOIN s1 ON gt.source1_entity_id = s1.entity_id
        JOIN s3 ON gt.target_id = s3.entity_id
        WHERE (norm_name(s1.business_name) = norm_name(s3.business_name) AND length(norm_name(s1.business_name)) >= 3)
           OR (sort_tokens(s1.business_name) = sort_tokens(s3.business_name) AND length(sort_tokens(s1.business_name)) >= 4)
           OR (first2_words(s1.business_name) != '' 
               AND first2_words(s1.business_name) = first2_words(s3.business_name)
               AND length(first2_words(s1.business_name)) >= 7)
           OR (addr_num(s1.business_address) != '' 
               AND addr_num(s1.business_address) = addr_num(s3.business_address)
               AND (substr(norm_name(s1.business_name), 1, 2) = substr(norm_name(s3.business_name), 1, 2)
                    OR substr(sort_tokens(s1.business_name), 1, 2) = substr(sort_tokens(s3.business_name), 1, 2)));
    """).fetchone()[0]

    print(f"S3 Combined Rule 1+2+3+4 matches: {s3_combined4:,} / {s3_total_gt:,} ({s3_combined4/s3_total_gt*100:.2f}%)")
    print(f"TOTAL COMBINED RECALL (Rules 1+2+3+4): {(s2_combined4 + s3_combined4) / total_gt * 100:.2f}%")

    print("\nSample pairs missed by both Rule 1 and Rule 2:")
    missed_both = con.execute("""
        SELECT s1.entity_id, s1.business_name, s1.business_address,
               s2.entity_id, s2.business_name, s2.business_address
        FROM gt
        JOIN s1 ON gt.source1_entity_id = s1.entity_id
        JOIN s2 ON gt.target_id = s2.entity_id
        WHERE norm_name(s1.business_name) != norm_name(s2.business_name)
          AND (addr_num(s1.business_address) = '' 
               OR addr_num(s1.business_address) != addr_num(s2.business_address)
               OR substr(norm_name(s1.business_name), 1, 2) != substr(norm_name(s2.business_name), 1, 2))
        LIMIT 10;
    """).fetchall()

    for r in missed_both:
        print(f"S1: [{r[0]}] {r[1]} | {r[2]}")
        print(f"S2: [{r[3]}] {r[4]} | {r[5]}")
        print("-" * 60)

if __name__ == "__main__":
    evaluate()

