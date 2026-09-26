#!/usr/bin/env python3
"""
Step 1: Deep Data Inspection & Profiling for ML Challenge 2026.
Stdlib-only implementation: streams data row by row, memory-safe, ultra-fast.
"""
import os
import sys
import csv
from collections import Counter, defaultdict

DELIM = "\t"

def inspect_source_file(file_path, file_label, max_samples=3):
    print(f"\n{'='*70}")
    print(f"FILE: {file_label} ({file_path})")
    print(f"{'='*70}")
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return None
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File Size: {file_size_mb:.2f} MB")
    
    total_rows = 0
    col_names = []
    missing_counts = defaultdict(int)
    unique_ids = set()
    dup_ids = 0
    countries = Counter()
    samples = []
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=DELIM)
        try:
            col_names = next(reader)
        except StopIteration:
            print("ERROR: Empty file")
            return None
        
        num_cols = len(col_names)
        print(f"Columns ({num_cols}): {col_names}")
        
        for row_idx, row in enumerate(reader):
            total_rows += 1
            if len(row) < num_cols:
                row.extend([""] * (num_cols - len(row)))
            
            entity_id = row[0].strip()
            if entity_id in unique_ids:
                dup_ids += 1
            else:
                unique_ids.add(entity_id)
            
            for i, col in enumerate(col_names):
                val = row[i].strip() if i < len(row) else ""
                if not val:
                    missing_counts[col] += 1
            
            # Country is usually index 3 or named 'country'
            if "country" in col_names:
                c_idx = col_names.index("country")
                if c_idx < len(row):
                    countries[row[c_idx].strip()] += 1
            
            if len(samples) < max_samples:
                samples.append(dict(zip(col_names, row[:num_cols])))
    
    print(f"Total Records / Rows: {total_rows:,}")
    print(f"Unique entity_ids: {len(unique_ids):,}")
    print(f"Duplicate entity_ids: {dup_ids:,}")
    
    print("\nMissing Values per Column:")
    for col in col_names:
        miss = missing_counts[col]
        pct = (miss / total_rows * 100) if total_rows > 0 else 0
        print(f"  - {col}: {miss:,} missing ({pct:.2f}%) | Non-null: {total_rows - miss:,}")
    
    if countries:
        print("\nCountry Distribution:")
        for country, count in countries.most_common():
            pct = count / total_rows * 100
            print(f"  - {country if country else '<EMPTY>'}: {count:,} ({pct:.2f}%)")
    
    print("\nSample Records:")
    for i, s in enumerate(samples, 1):
        print(f"  Sample {i}:")
        for k, v in s.items():
            print(f"    {k}: {v[:80] + ('...' if len(v) > 80 else '')}")
            
    return {
        "file": file_label,
        "rows": total_rows,
        "unique_ids": len(unique_ids),
        "columns": col_names,
        "countries": dict(countries),
    }

def inspect_ground_truth(file_path):
    print(f"\n{'='*70}")
    print(f"GROUND TRUTH: {file_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(file_path):
        print("ERROR: Ground truth file not found")
        return None
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"File Size: {file_size_mb:.2f} MB")
    
    total_s1 = 0
    singletons = 0  # 0 matches
    match_count_dist = Counter()
    source2_matches = 0
    source3_matches = 0
    unique_s2_matched = set()
    unique_s3_matched = set()
    samples = []
    
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=DELIM)
        header = next(reader, None)
        print(f"Header: {header}")
        
        for row_idx, row in enumerate(reader):
            total_s1 += 1
            s1_id = row[0].strip() if len(row) > 0 else ""
            matches_str = row[1].strip() if len(row) > 1 else ""
            
            if not matches_str:
                singletons += 1
                match_count_dist[0] += 1
                if len(samples) < 5 and match_count_dist[0] <= 2:
                    samples.append((s1_id, "EMPTY (Singleton)"))
                continue
            
            matched_ids = [m.strip() for m in matches_str.split(",") if m.strip()]
            num_matches = len(matched_ids)
            match_count_dist[num_matches] += 1
            
            for mid in matched_ids:
                if mid.startswith("S2-"):
                    source2_matches += 1
                    unique_s2_matched.add(mid)
                elif mid.startswith("S3-"):
                    source3_matches += 1
                    unique_s3_matched.add(mid)
            
            if len(samples) < 5:
                samples.append((s1_id, matches_str))
                
    print(f"Total Source 1 records in GT: {total_s1:,}")
    print(f"Singletons (0 matches): {singletons:,} ({singletons / total_s1 * 100:.2f}%)")
    print(f"Entities with >= 1 match: {total_s1 - singletons:,} ({(total_s1 - singletons) / total_s1 * 100:.2f}%)")
    print(f"Total Matched S2 records: {source2_matches:,} (Unique S2: {len(unique_s2_matched):,})")
    print(f"Total Matched S3 records: {source3_matches:,} (Unique S3: {len(unique_s3_matched):,})")
    print(f"Total Matched links: {source2_matches + source3_matches:,}")
    
    print("\nMatch Count Distribution per S1 Entity:")
    for num_m in sorted(match_count_dist.keys())[:10]:
        count = match_count_dist[num_m]
        pct = count / total_s1 * 100
        print(f"  - {num_m} matches: {count:,} entities ({pct:.2f}%)")
    if len(match_count_dist) > 10:
        remaining = sum(count for k, count in match_count_dist.items() if k >= 10)
        print(f"  - 10+ matches: {remaining:,} entities ({remaining / total_s1 * 100:.2f}%)")
    print(f"Max matches for a single entity: {max(match_count_dist.keys())}")
    
    print("\nSample Ground Truth mappings:")
    for s1, m in samples:
        print(f"  {s1} -> {m}")

def main():
    base_dir = "/Users/hexahealth/Desktop/ML"
    print("Starting Step 1: Deep Data Inspection...\n")
    
    train_files = [
        ("train_source1.tsv", "Train Source 1"),
        ("train_source2.tsv", "Train Source 2"),
        ("train_source3.tsv", "Train Source 3"),
    ]
    for fname, label in train_files:
        inspect_source_file(os.path.join(base_dir, "dataset/train", fname), label)
        
    inspect_ground_truth(os.path.join(base_dir, "dataset/train/train_ground_truth.tsv"))
    
    test_files = [
        ("test_source1.tsv", "Test Source 1"),
        ("test_source2.tsv", "Test Source 2"),
        ("test_source3.tsv", "Test Source 3"),
    ]
    for fname, label in test_files:
        inspect_source_file(os.path.join(base_dir, "dataset/test", fname), label)

if __name__ == "__main__":
    main()
