#!/usr/bin/env python3
"""
src/build_test_cache.py
Precomputes and caches normalized features for test_source1, test_source2, and test_source3.
"""
import os
import sys
import time
import multiprocessing
multiprocessing.set_start_method('fork', force=True)
from multiprocessing import Pool
import duckdb
import polars as pl

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(base_dir, 'src'))
import cleaner_v6 as cv6

def clean_file(name, in_fname, out_fname):
    out_path = os.path.join(base_dir, '.tmp', out_fname)
    if os.path.exists(out_path):
        print(f"✓ Cache already exists: {out_path} ({os.path.getsize(out_path)/(1024*1024):.1f} MB)")
        return
        
    print(f"\nProcessing {name} ({in_fname})...")
    t0 = time.time()
    in_path = os.path.join(base_dir, 'dataset', 'test', in_fname)
    
    con = duckdb.connect()
    con.execute("PRAGMA threads=8; PRAGMA memory_limit='10GB';")
    rows = con.execute(f"""
        SELECT entity_id, business_name, business_address, trim(country) as country 
        FROM read_csv('{in_path}', delim='\\t', header=true)
    """).fetchall()
    print(f"  ✓ Loaded {len(rows):,} rows in {time.time()-t0:.1f}s. Normalizing across 8 CPU cores...")
    
    t1 = time.time()
    with Pool(8) as p:
        records = p.map(cv6.clean_single_record, rows, chunksize=10000)
    print(f"  ✓ Cleaned {len(records):,} records in {time.time()-t1:.1f}s ({len(records)/(time.time()-t1):.0f} rows/s)")
    
    t2 = time.time()
    cols = ['entity_id', 'cname', 'caddr', 'country', 'nname', 'stoks', 'fcw', 'pfx4',
            'num1', 'num2', 'w1', 'w2', 'w3', 'fka_nname', 'fka_stoks', 'fka_fcw']
    df = pl.DataFrame({c: [r[i] for r in records] for i, c in enumerate(cols)})
    df.write_parquet(out_path)
    print(f"  ✓ Saved to {out_path} ({os.path.getsize(out_path)/(1024*1024):.1f} MB) in {time.time()-t2:.1f}s")

if __name__ == '__main__':
    os.makedirs(os.path.join(base_dir, '.tmp'), exist_ok=True)
    t_start = time.time()
    print("=" * 70)
    print("🚀 BUILDING HIGH-SPEED TEST PARQUET CACHES FOR AMAZON ML CHALLENGE")
    print("=" * 70)
    clean_file('Test S1', 'test_source1.tsv', 'test_s1_clean_v6.parquet')
    clean_file('Test S2', 'test_source2.tsv', 'test_s2_clean_v6.parquet')
    clean_file('Test S3', 'test_source3.tsv', 'test_s3_clean_v6.parquet')
    print(f"\nAll Test Caches built in {time.time() - t_start:.1f}s!")
