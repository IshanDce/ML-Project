#!/usr/bin/env python3
"""
src/measure_true_recall.py
Accurately measure candidate blocking recall across the ENTIRE Source 2 and Source 3 files
for a representative slice of 2,000 S1 entities.
"""
import os
import sys
import csv
import time
from collections import defaultdict
from cleaner import clean_business_name, clean_address

DELIM = "\t"

def test_recall(num_s1=2000, country="US"):
    print(f"Loading {num_s1} S1 entities for country={country}...")
    s1_dict = {}
    with open("dataset/train/train_source1.tsv", "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter=DELIM)
        next(r)
        for row in r:
            if row[3] == country:
                s1_dict[row[0]] = (row[1], row[2])
                if len(s1_dict) >= num_s1:
                    break
                    
    target_s1_ids = set(s1_dict.keys())
    
    # Load GT
    gt = defaultdict(set)
    total_gt = 0
    with open("dataset/train/train_ground_truth.tsv", "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter=DELIM)
        next(r)
        for row in r:
            if row[0] in target_s1_ids and len(row) > 1 and row[1]:
                mids = [m.strip() for m in row[1].split(",") if m.strip()]
                for m in mids:
                    gt[row[0]].add(m)
                    total_gt += 1
                    
    print(f"Total True Matches for these {len(target_s1_ids)} entities: {total_gt}")
    
    # Preprocess S1
    s1_info = {}
    exact_core_idx = defaultdict(list)
    exact_nosp_idx = defaultdict(list)
    first_tok_idx = defaultdict(list)
    rare_tok_idx = defaultdict(list)
    
    # Token frequencies across these S1 entities
    tok_freq = defaultdict(int)
    for s1_id, (name, addr) in s1_dict.items():
        c_name, core_name, f_tok = clean_business_name(name)
        c_addr, nums = clean_address(addr)
        addr_toks = set(c_addr.split())
        c_toks = set(core_name.split())
        for t in c_toks:
            if len(t) >= 4:
                tok_freq[t] += 1
        s1_info[s1_id] = {
            "core": core_name,
            "nosp": c_name.replace(" ", ""),
            "ftok": f_tok,
            "c_toks": c_toks,
            "addr_toks": addr_toks,
            "nums": nums
        }
        
    for s1_id, d in s1_info.items():
        if d["core"]:
            exact_core_idx[d["core"]].append(s1_id)
        if len(d["nosp"]) >= 4:
            exact_nosp_idx[d["nosp"]].append(s1_id)
        if len(d["ftok"]) >= 3:
            first_tok_idx[d["ftok"]].append(s1_id)
        for t in d["c_toks"]:
            if len(t) >= 4 and tok_freq[t] <= 100:
                rare_tok_idx[t].append(s1_id)
                
    candidates = defaultdict(set)
    missed_gt_examples = []
    
    def scan_file(filepath, label):
        print(f"Scanning FULL {label}...")
        t0 = time.time()
        count = 0
        with open(filepath, "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter=DELIM)
            next(r)
            for row in r:
                if row[3] != country:
                    continue
                count += 1
                s2_id = row[0]
                name = row[1]
                addr = row[2]
                
                c_name, core_name, f_tok = clean_business_name(name)
                c_addr, nums = clean_address(addr)
                addr_toks = set(c_addr.split())
                nosp = c_name.replace(" ", "")
                c_toks = set(core_name.split())
                
                matched_s1 = set()
                
                if core_name and core_name in exact_core_idx:
                    matched_s1.update(exact_core_idx[core_name])
                    
                if len(nosp) >= 4 and nosp in exact_nosp_idx:
                    matched_s1.update(exact_nosp_idx[nosp])
                    
                # Rare token
                for t in c_toks:
                    if t in rare_tok_idx:
                        for s1_id in rare_tok_idx[t]:
                            s1 = s1_info[s1_id]
                            # if token is distinct (len >= 6) OR share address/number
                            if len(t) >= 6 or (s1["addr_toks"] & addr_toks) or (s1["nums"] & nums):
                                matched_s1.add(s1_id)
                                
                # First token
                if len(f_tok) >= 3 and f_tok in first_tok_idx:
                    for s1_id in first_tok_idx[f_tok]:
                        s1 = s1_info[s1_id]
                        if (s1["addr_toks"] & addr_toks) or (s1["nums"] & nums):
                            matched_s1.add(s1_id)
                            
                for s1_id in matched_s1:
                    candidates[s1_id].add(s2_id)
                    
        print(f"Scanned {count:,} records in {time.time() - t0:.2f}s")
        
    scan_file("dataset/train/train_source2.tsv", "Train Source 2")
    scan_file("dataset/train/train_source3.tsv", "Train Source 3")
    
    # Calculate exact recall
    captured = 0
    for s1_id in target_s1_ids:
        cands = candidates[s1_id]
        for true_id in gt[s1_id]:
            if true_id in cands:
                captured += 1
            else:
                if len(missed_gt_examples) < 10:
                    missed_gt_examples.append((s1_id, true_id))
                    
    recall = (captured / total_gt * 100) if total_gt > 0 else 0
    total_cands = sum(len(c) for c in candidates.values())
    avg_cands = total_cands / len(target_s1_ids)
    
    print("\n" + "="*60)
    print("--- FULL SCAN RECALL RESULTS ---")
    print(f"Target S1 entities:            {len(target_s1_ids):,}")
    print(f"Total True Matches:            {total_gt:,}")
    print(f"Captured Matches:              {captured:,}")
    print(f"BLOCKING RECALL:               {recall:.2f}%")
    print(f"Average candidates per S1:     {avg_cands:.2f}")
    print("="*60)
    
    if missed_gt_examples:
        print("\nInvestigating sample missed true matches:")
        # Look up missing records
        missed_s2_ids = {m for _, m in missed_gt_examples}
        missed_records = {}
        with open("dataset/train/train_source2.tsv") as f:
            r = csv.reader(f, delimiter=DELIM)
            next(r)
            for row in r:
                if row[0] in missed_s2_ids:
                    missed_records[row[0]] = (row[1], row[2])
        with open("dataset/train/train_source3.tsv") as f:
            r = csv.reader(f, delimiter=DELIM)
            next(r)
            for row in r:
                if row[0] in missed_s2_ids:
                    missed_records[row[0]] = (row[1], row[2])
                    
        for s1_id, m_id in missed_gt_examples[:5]:
            print(f"\n[MISSED] S1 ID: {s1_id}")
            print(f"  S1: Name:    {s1_dict[s1_id][0]}")
            print(f"      Address: {s1_dict[s1_id][1]}")
            if m_id in missed_records:
                print(f"  Target {m_id}:")
                print(f"      Name:    {missed_records[m_id][0]}")
                print(f"      Address: {missed_records[m_id][1]}")

if __name__ == "__main__":
    test_recall(num_s1=2000, country="US")
