#!/usr/bin/env python3
"""
src/eval_india_blocking.py
Evaluate candidate blocking on India training data with Devanagari transliteration.
"""
import duckdb
import time
from cleaner import transliterate_devanagari, strip_accents, clean_business_name

def evaluate_india():
    t0 = time.time()
    con = duckdb.connect()
    
    print("Evaluating Candidate Blocking on India (10,000 S1 records)...")
    
    con.create_function("translit", lambda s: transliterate_devanagari(s) if s else "", ["VARCHAR"], "VARCHAR")
    
    # 2. Load 10,000 S1 India records
    con.execute("""
        CREATE TABLE s1 AS 
        SELECT entity_id, translit(business_name) AS business_name, translit(business_address) AS business_address, country
        FROM read_csv('dataset/train/train_source1.tsv', delim='\t', header=true)
        WHERE country = 'India'
        LIMIT 10000;
    """)
    
    # 3. Load Ground Truth
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
    
    # 4. Load S2 and S3 for India
    print("Loading S2 and S3 India...")
    con.execute("""
        CREATE TABLE s2 AS 
        SELECT entity_id, translit(business_name) AS business_name, translit(business_address) AS business_address 
        FROM read_csv('dataset/train/train_source2.tsv', delim='\t', header=true)
        WHERE country = 'India';
    """)
    con.execute("""
        CREATE TABLE s3 AS 
        SELECT entity_id, translit(business_name) AS business_name, translit(business_address) AS business_address 
        FROM read_csv('dataset/train/train_source3.tsv', delim='\t', header=true)
        WHERE country = 'India';
    """)
    print(f"Loaded in {time.time() - t0:.2f}s")
    
    # 5. Define normalization macros
    con.execute(r"""
        CREATE MACRO norm_name(s) AS (
            regexp_replace(
                replace(
                    regexp_replace(
                        regexp_replace(
                            trim(regexp_replace(lower(replace(s, '&', ' and ')), '[^a-z0-9]+', ' ', 'g')),
                            '\b(pvt|private|ltd|limited|llp|corp|corporation|inc|co|company|enterprises|services|solutions|systems|impex|industries)\b', '', 'g'
                        ),
                        '\b(com|org|net|in|co\.in)\b', '', 'g'
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
                                    '\b(the|and|pvt|private|ltd|limited|llp|co|company)\b', '', 'g'
                                ),
                                '\b(com|org|net|in)\b', '', 'g'
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
            ltrim(regexp_extract(s, '\b([0-9]{1,7})\b', 1), '0')
        );
    """)
    
    # 6. Evaluate combined recall on India S2
    s2_combined = con.execute("""
        SELECT count(DISTINCT gt.target_id)
        FROM gt
        JOIN s1 ON gt.source1_entity_id = s1.entity_id
        JOIN s2 ON gt.target_id = s2.entity_id
        WHERE (norm_name(s1.business_name) = norm_name(s2.business_name) AND length(norm_name(s1.business_name)) >= 3)
           OR (sort_tokens(s1.business_name) = sort_tokens(s2.business_name) AND length(sort_tokens(s1.business_name)) >= 4)
           OR (addr_num(s1.business_address) != '' 
               AND addr_num(s1.business_address) = addr_num(s2.business_address)
               AND (substr(norm_name(s1.business_name), 1, 2) = substr(norm_name(s2.business_name), 1, 2)
                    OR substr(sort_tokens(s1.business_name), 1, 2) = substr(sort_tokens(s2.business_name), 1, 2)));
    """).fetchone()[0]
    
    # 7. Evaluate combined recall on India S3
    s3_combined = con.execute("""
        SELECT count(DISTINCT gt.target_id)
        FROM gt
        JOIN s1 ON gt.source1_entity_id = s1.entity_id
        JOIN s3 ON gt.target_id = s3.entity_id
        WHERE (norm_name(s1.business_name) = norm_name(s3.business_name) AND length(norm_name(s1.business_name)) >= 3)
           OR (sort_tokens(s1.business_name) = sort_tokens(s3.business_name) AND length(sort_tokens(s1.business_name)) >= 4)
           OR (addr_num(s1.business_address) != '' 
               AND addr_num(s1.business_address) = addr_num(s3.business_address)
               AND (substr(norm_name(s1.business_name), 1, 2) = substr(norm_name(s3.business_name), 1, 2)
                    OR substr(sort_tokens(s1.business_name), 1, 2) = substr(sort_tokens(s3.business_name), 1, 2)));
    """).fetchone()[0]
    
    print(f"\nIndia S2 Combined Recall: {s2_combined:,} / {s2_total_gt:,} ({s2_combined/s2_total_gt*100:.2f}%)")
    print(f"India S3 Combined Recall: {s3_combined:,} / {s3_total_gt:,} ({s3_combined/s3_total_gt*100:.2f}%)")
    print(f"India TOTAL COMBINED RECALL: {(s2_combined + s3_combined) / total_gt * 100:.2f}%")
    print(f"Elapsed Time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    evaluate_india()
