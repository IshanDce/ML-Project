#!/usr/bin/env python3
"""
src/test_blocking_recall.py
Evaluate candidate generation / blocking strategy recall & reduction ratio on Train Ground Truth.
"""
import os
import sys
import csv
import time
from collections import defaultdict
from cleaner import clean_business_name, clean_address, strip_accents

DELIM = "\t"

def run_blocking_benchmark(sample_size=30000, country_filter="US"):
    print(f"\n--- BENCHMARKING BLOCKING STRATEGY ON {country_filter} ({sample_size:,} S1 entities) ---")
    start_time = time.time()
    
    # 1. Load sample S1 records
    s1_dict = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter=DELIM)
        next(r)
        for row in r:
            if row[3] == country_filter:
                s1_dict[row[0]] = (row[1], row[2])
                if len(s1_dict) >= sample_size:
                    break
                    
    target_s1_ids = set(s1_dict.keys())
    print(f"Loaded {len(target_s1_ids):,} S1 entities in {time.time() - start_time:.2f}s")
    
    # 2. Load Ground Truth for these S1 records
    gt_pairs = defaultdict(set)
    total_true_matches = 0
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter=DELIM)
        next(r)
        for row in r:
            s1_id = row[0]
            if s1_id in target_s1_ids and len(row) > 1 and row[1]:
                mids = [m.strip() for m in row[1].split(",") if m.strip()]
                for m in mids:
                    gt_pairs[s1_id].add(m)
                    total_true_matches += 1
                    
    print(f"Total True Matches in Ground Truth: {total_true_matches:,}")
    
    # 3. Clean and Index S1 records
    s1_preprocessed = {}
    for s1_id, (name, addr) in s1_dict.items():
        c_name, core_name, f_tok = clean_business_name(name)
        c_addr, nums = clean_address(addr)
        addr_tokens = set(c_addr.split())
        name_no_space = c_name.replace(" ", "")
        s1_preprocessed[s1_id] = {
            "clean_name": c_name,
            "core_name": core_name,
            "first_tok": f_tok,
            "name_no_space": name_no_space,
            "addr_tokens": addr_tokens,
            "nums": nums,
            "core_tokens": set(core_name.split())
        }
        
    # Build Inverted Indexes for S1 to match streaming S2 & S3 against S1!
    # Matching streaming S2/S3 against indexed S1 uses MUCH LESS RAM!
    exact_core_idx = defaultdict(list)
    exact_nosp_idx = defaultdict(list)
    first_tok_idx = defaultdict(list)
    rare_token_idx = defaultdict(list)
    
    # Count token frequencies in S1 to find rare tokens
    tok_freq = defaultdict(int)
    for s1_id, data in s1_preprocessed.items():
        for tok in data["core_tokens"]:
            if len(tok) >= 4:
                tok_freq[tok] += 1
                
    for s1_id, data in s1_preprocessed.items():
        exact_core_idx[data["core_name"]].append(s1_id)
        if len(data["name_no_space"]) >= 4:
            exact_nosp_idx[data["name_no_space"]].append(s1_id)
        if len(data["first_tok"]) >= 3:
            first_tok_idx[data["first_tok"]].append(s1_id)
        for tok in data["core_tokens"]:
            if len(tok) >= 4 and tok_freq[tok] <= 200:
                rare_token_idx[tok].append(s1_id)
                
    print(f"S1 Indexing complete. Indexed {len(exact_core_idx):,} core names, {len(first_tok_idx):,} first tokens.")
    
    # 4. Stream S2 and S3, generating candidates for S1
    candidates = defaultdict(set)
    
    def process_source(filepath, src_label, max_rows=500000):
        print(f"Streaming {src_label}...")
        t0 = time.time()
        checked_rows = 0
        with open(filepath, "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter=DELIM)
            next(r)
            for row in r:
                if row[3] != country_filter:
                    continue
                checked_rows += 1
                if checked_rows > max_rows:
                    break
                    
                s2_id = row[0]
                name = row[1]
                addr = row[2]
                
                c_name, core_name, f_tok = clean_business_name(name)
                c_addr, nums = clean_address(addr)
                addr_tokens = set(c_addr.split())
                name_nosp = c_name.replace(" ", "")
                core_tokens = set(core_name.split())
                
                matched_s1s = set()
                
                # Rule 1: Exact core name match
                if core_name in exact_core_idx:
                    matched_s1s.update(exact_core_idx[core_name])
                    
                # Rule 2: No-space name match (handles spacing differences & URLs)
                if len(name_nosp) >= 4 and name_nosp in exact_nosp_idx:
                    matched_s1s.update(exact_nosp_idx[name_nosp])
                    
                # Rule 3: Rare token match with any address overlap or number overlap
                for tok in core_tokens:
                    if tok in rare_token_idx:
                        for cand_s1 in rare_token_idx[tok]:
                            # check if at least 1 address token or number matches, or token is very long (>=6)
                            s1_info = s1_preprocessed[cand_s1]
                            if len(tok) >= 6 or (s1_info["addr_tokens"] & addr_tokens) or (s1_info["nums"] & nums):
                                matched_s1s.add(cand_s1)
                                
                # Rule 4: First token match + address / number overlap
                if len(f_tok) >= 3 and f_tok in first_tok_idx:
                    for cand_s1 in first_tok_idx[f_tok]:
                        s1_info = s1_preprocessed[cand_s1]
                        # Must share at least one address token or number
                        if (s1_info["addr_tokens"] & addr_tokens) or (s1_info["nums"] & nums):
                            matched_s1s.add(cand_s1)
                            
                for cand_s1 in matched_s1s:
                    if len(candidates[cand_s1]) < 40:
                        candidates[cand_s1].add(s2_id)
                        
        print(f"Processed {checked_rows:,} rows of {src_label} in {time.time() - t0:.2f}s")
        
    process_source("dataset/train/train_source2.tsv", "Train Source 2 (sample)")
    process_source("dataset/train/train_source3.tsv", "Train Source 3 (sample)")
    
    # 5. Measure Recall on Ground Truth
    captured_matches = 0
    total_relevant_gt = 0
    
    # Only evaluate on GT matches that were present in the sampled S2/S3 rows
    # Let's count how many GT pairs were captured
    total_candidates_generated = 0
    for s1_id in target_s1_ids:
        cands = candidates[s1_id]
        total_candidates_generated += len(cands)
        for true_id in gt_pairs[s1_id]:
            # Check if true_id was captured
            if true_id in cands:
                captured_matches += 1
            total_relevant_gt += 1
            
    avg_cands = total_candidates_generated / len(target_s1_ids) if target_s1_ids else 0
    recall = (captured_matches / total_relevant_gt * 100) if total_relevant_gt > 0 else 0
    
    print("\n" + "="*60)
    print("--- BLOCKING BENCHMARK RESULTS ---")
    print(f"Target S1 entities:            {len(target_s1_ids):,}")
    print(f"Average candidates per S1:     {avg_cands:.2f}")
    print(f"Total True Matches in Sample:  {total_relevant_gt:,}")
    print(f"Captured True Matches:         {captured_matches:,}")
    print(f"Candidate Blocking Recall:     {recall:.2f}%")
    print(f"Total Benchmark Time:          {time.time() - start_time:.2f}s")
    print("="*60)

if __name__ == "__main__":
    run_blocking_benchmark(sample_size=10000, country_filter="US")
