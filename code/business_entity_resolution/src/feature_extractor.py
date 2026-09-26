#!/usr/bin/env python3
"""
src/feature_extractor.py
High-Performance Feature Extraction for Entity Resolution Pairs using RapidFuzz & Polars.
"""
import re
from rapidfuzz import fuzz, distance

# Common address token stopwords
ADDR_STOPWORDS = {'st', 'street', 'rd', 'road', 'ave', 'avenue', 'dr', 'drive', 'blvd', 'boulevard', 
                  'hwy', 'highway', 'ct', 'court', 'ln', 'lane', 'fl', 'floor', 'ste', 'suite', 
                  'unit', 'apt', 'apartment', 'po', 'box', 'city', 'town'}

def extract_numbers(s):
    if not s:
        return set()
    return set(re.findall(r'\b\d+\b', s))

def compute_pair_features(s1_name, s1_addr, cand_name, cand_addr, is_s2=1, channel_priority=1):
    """
    Computes a vector of numerical similarity features for a candidate pair.
    Returns: list of float values.
    """
    s1_name = s1_name or ""
    cand_name = cand_name or ""
    s1_addr = s1_addr or ""
    cand_addr = cand_addr or ""
    
    # 1. Name String Metrics (RapidFuzz C++ speed)
    lev_ratio = fuzz.ratio(s1_name, cand_name) / 100.0
    jw_sim = distance.JaroWinkler.similarity(s1_name, cand_name)
    tok_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
    tok_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
    partial = fuzz.partial_ratio(s1_name, cand_name) / 100.0
    
    # Length diff
    l1, l2 = len(s1_name), len(cand_name)
    len_diff = abs(l1 - l2) / max(l1, l2, 1)
    
    # First token match
    w1 = s1_name.split()[0] if s1_name.split() else ""
    w2 = cand_name.split()[0] if cand_name.split() else ""
    first_tok_match = 1.0 if (w1 and w2 and w1 == w2) else 0.0
    
    # Exact normalized name match
    exact_match = 1.0 if (s1_name and cand_name and s1_name == cand_name) else 0.0
    
    # 2. Address Metrics
    addr_tok_sort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
    
    s1_addr_toks = set(s1_addr.split()) - ADDR_STOPWORDS
    c_addr_toks = set(cand_addr.split()) - ADDR_STOPWORDS
    
    if s1_addr_toks and c_addr_toks:
        addr_jaccard = len(s1_addr_toks & c_addr_toks) / len(s1_addr_toks | c_addr_toks)
    else:
        addr_jaccard = 0.0
        
    # Number overlap (street number / zip)
    s1_nums = extract_numbers(s1_addr)
    c_nums = extract_numbers(cand_addr)
    
    if not s1_nums or not c_nums:
        num_match = 0.0  # missing numbers
    elif s1_nums & c_nums:
        num_match = 1.0  # exact number match
    else:
        num_match = -1.0 # conflicting numbers
        
    return [
        lev_ratio,
        jw_sim,
        tok_sort,
        tok_set,
        partial,
        len_diff,
        first_tok_match,
        exact_match,
        addr_tok_sort,
        addr_jaccard,
        num_match,
        float(is_s2),
        float(channel_priority)
    ]

FEATURE_NAMES = [
    "lev_ratio",
    "jw_sim",
    "tok_sort",
    "tok_set",
    "partial",
    "len_diff",
    "first_tok_match",
    "exact_match",
    "addr_tok_sort",
    "addr_jaccard",
    "num_match",
    "is_s2",
    "channel_priority"
]

if __name__ == "__main__":
    # Test feature extraction on actual match vs non-match
    f1 = compute_pair_features("strategic praetorian", "702 n st washington dc",
                               "strategic praetorian inc", "n st washington dc", is_s2=1, channel_priority=1)
    f2 = compute_pair_features("strategic praetorian", "702 n st washington dc",
                               "global motors llc", "100 main st chicago il", is_s2=1, channel_priority=4)
    print("Match Features:    ", [round(x, 3) for x in f1])
    print("Non-Match Features:", [round(x, 3) for x in f2])
