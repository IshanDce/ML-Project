#!/usr/bin/env python3
"""
src/train_model.py
End-to-End Model Training & F0.5 Metric Optimization for Business Entity Resolution.
Trains LightGBM classifier on realistic positive & negative pairs, optimizes threshold on holdout validation.
"""
import os
import sys
import time
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import duckdb

from feature_extractor import compute_pair_features, FEATURE_NAMES

def calculate_macro_f05(gt_mapping, pred_mapping, s1_eval_ids):
    """
    Computes official macro-averaged F0.5 score across all evaluation S1 entities.
    Handles singletons (empty ground truth) strictly per competition rules.
    """
    scores = []
    for s1_id in s1_eval_ids:
        true_set = gt_mapping.get(s1_id, set())
        pred_set = pred_mapping.get(s1_id, set())
        
        # Singleton logic
        if not true_set and not pred_set:
            scores.append(1.0)
            continue
        if not true_set and pred_set:
            scores.append(0.0)
            continue
        if true_set and not pred_set:
            scores.append(0.0)
            continue
            
        # Non-empty ground truth
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if prec == 0.0 or rec == 0.0:
            scores.append(0.0)
        else:
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            scores.append(f05)
            
    return np.mean(scores)

def build_training_dataset(num_s1=30000):
    t0 = time.time()
    print(f"Building training and validation dataset from {num_s1:,} S1 entities...")
    con = duckdb.connect()
    
    # 1. Load S1 sample
    con.execute(f"""
        CREATE TABLE s1 AS 
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('dataset/train/train_source1.tsv', delim='\t', header=true)
        LIMIT {num_s1};
    """)
    
    # 2. Load Ground Truth for S1
    con.execute("""
        CREATE TABLE gt AS
        SELECT source1_entity_id as s1_id, unnest(string_split(matched_entity_ids, ',')) AS target_id
        FROM read_csv('dataset/train/train_ground_truth.tsv', delim='\t', header=true)
        WHERE source1_entity_id IN (SELECT entity_id FROM s1) AND matched_entity_ids != '';
    """)
    
    # 3. Load S2 and S3 for these countries
    con.execute("""
        CREATE TABLE s2 AS 
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('dataset/train/train_source2.tsv', delim='\t', header=true)
        WHERE entity_id IN (SELECT target_id FROM gt)
           OR (entity_id IN (
               SELECT entity_id FROM read_csv('dataset/train/train_source2.tsv', delim='\t', header=true) LIMIT 100000
           ));
           
        CREATE TABLE s3 AS 
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('dataset/train/train_source3.tsv', delim='\t', header=true)
        WHERE entity_id IN (SELECT target_id FROM gt)
           OR (entity_id IN (
               SELECT entity_id FROM read_csv('dataset/train/train_source3.tsv', delim='\t', header=true) LIMIT 100000
           ));
    """)
    
    print(f"Base data loaded in {time.time() - t0:.2f}s")
    
    # 4. Extract Positive Pairs (Ground Truth)
    print("Extracting positive pairs...")
    pos_pairs = con.execute("""
        SELECT s1.entity_id as s1_id, s1.business_name as s1_name, s1.business_address as s1_addr,
               target.entity_id as target_id, target.business_name as cand_name, target.business_address as cand_addr,
               CASE WHEN target.entity_id LIKE 'S2-%' THEN 1 ELSE 0 END as is_s2,
               1 as priority, 1 as label
        FROM gt
        JOIN s1 ON gt.s1_id = s1.entity_id
        JOIN (
            SELECT entity_id, business_name, business_address FROM s2
            UNION ALL
            SELECT entity_id, business_name, business_address FROM s3
        ) target ON gt.target_id = target.entity_id;
    """).fetchall()
    
    print(f"Extracted {len(pos_pairs):,} positive pairs.")
    
    # 5. Extract Hard Negative Pairs (Candidates from blocking that are NOT in GT)
    print("Extracting hard negative pairs via candidate generation...")
    con.execute(r"""
        CREATE MACRO norm_name(s) AS (
            regexp_replace(replace(regexp_replace(regexp_replace(trim(regexp_replace(lower(replace(s, '&', ' and ')), '[^a-z0-9]+', ' ', 'g')), '\b(inc|corp|corporation|incorporated|llc|llp|ltd|limited|co|company|enterprises|pvt)\b', '', 'g'), '\b(com|org|net)\b', '', 'g'), ' ', ''), '[^a-z0-9]', '', 'g')
        );
        CREATE MACRO addr_num(s) AS (
            ltrim(regexp_extract(s, '\b([0-9]{1,7})\b', 1), '0')
        );
        
        CREATE TABLE raw_neg AS
        SELECT s1.entity_id as s1_id, s1.business_name as s1_name, s1.business_address as s1_addr,
               s2.entity_id as target_id, s2.business_name as cand_name, s2.business_address as cand_addr,
               1 as is_s2, 2 as priority, 0 as label
        FROM s1
        JOIN s2 ON norm_name(s1.business_name) = norm_name(s2.business_name) AND s1.country = s2.country
        WHERE NOT EXISTS (SELECT 1 FROM gt WHERE gt.s1_id = s1.entity_id AND gt.target_id = s2.entity_id)
        LIMIT 60000;
    """)
    neg_pairs = con.execute("SELECT * FROM raw_neg").fetchall()
    print(f"Extracted {len(neg_pairs):,} negative pairs.")
    
    # Combine positive and negative pairs
    all_pairs = pos_pairs + neg_pairs
    print(f"Total pairs for training/validation: {len(all_pairs):,}")
    
    # 6. Extract features for all pairs
    print("Computing similarity feature vectors...")
    X = []
    y = []
    s1_ids = []
    cand_ids = []
    
    for row in all_pairs:
        s1_id, s1_name, s1_addr, cand_id, cand_name, cand_addr, is_s2, pri, label = row
        feats = compute_pair_features(s1_name, s1_addr, cand_name, cand_addr, is_s2, pri)
        X.append(feats)
        y.append(label)
        s1_ids.append(s1_id)
        cand_ids.append(cand_id)
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    s1_ids = np.array(s1_ids)
    cand_ids = np.array(cand_ids)
    
    print(f"Feature matrix shape: {X.shape}, Positives: {sum(y == 1):,}, Negatives: {sum(y == 0):,}")
    return X, y, s1_ids, cand_ids

def train_and_optimize():
    X, y, s1_ids, cand_ids = build_training_dataset(num_s1=35000)
    
    # Unique S1 entities for group split
    unique_s1 = np.unique(s1_ids)
    train_s1, val_s1 = train_test_split(unique_s1, test_size=0.25, random_state=42)
    train_s1_set = set(train_s1)
    val_s1_set = set(val_s1)
    
    train_mask = np.isin(s1_ids, train_s1)
    val_mask = np.isin(s1_ids, val_s1)
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_s1_ids = s1_ids[val_mask]
    val_cand_ids = cand_ids[val_mask]
    
    print(f"\nTrain set: {len(X_train):,} pairs ({len(train_s1):,} S1 entities)")
    print(f"Val set:   {len(X_val):,} pairs ({len(val_s1):,} S1 entities)")
    
    # Train LightGBM model
    print("\nTraining LightGBM Classifier...")
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 31,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42
    }
    
    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.record_evaluation(evals_result), lgb.early_stopping(stopping_rounds=25, verbose=False)]
    )
    
    print("Training finished!")
    
    # Feature Importances
    importances = model.feature_importance(importance_type='gain')
    sorted_idx = np.argsort(importances)[::-1]
    print("\nTop 5 Most Important Features (Gain):")
    for idx in sorted_idx[:5]:
        print(f"  - {FEATURE_NAMES[idx]}: {importances[idx]:.1f}")
        
    # Validation Predictions
    val_probs = model.predict(X_val)
    
    # Load Ground Truth for Validation S1 entities
    con = duckdb.connect()
    gt_df = con.execute("""
        SELECT source1_entity_id, matched_entity_ids
        FROM read_csv('dataset/train/train_ground_truth.tsv', delim='\t', header=true);
    """).fetchall()
    
    gt_mapping = {}
    for s1_id, m_str in gt_df:
        if s1_id in val_s1_set:
            if m_str:
                gt_mapping[s1_id] = set(m.strip() for m in m_str.split(",") if m.strip())
            else:
                gt_mapping[s1_id] = set()
            
    # Search for optimal threshold for F0.5
    print("\nOptimizing Threshold specifically for Macro F0.5 Metric...")
    best_f05 = -1.0
    best_thresh = 0.50
    
    thresholds = np.arange(0.20, 0.96, 0.05)
    for thresh in thresholds:
        pred_mapping = {s1: set() for s1 in val_s1_set}
        for s1_id, c_id, prob in zip(val_s1_ids, val_cand_ids, val_probs):
            if prob >= thresh:
                pred_mapping[s1_id].add(c_id)
                
        f05_score = calculate_macro_f05(gt_mapping, pred_mapping, val_s1_set)
        print(f"  Threshold = {thresh:.2f} --> Validation Macro F0.5: {f05_score:.4f}")
        if f05_score > best_f05:
            best_f05 = f05_score
            best_thresh = thresh
            
    print("\n" + "="*60)
    print(f"OPTIMAL THRESHOLD:       {best_thresh:.2f}")
    print(f"PEAK VALIDATION F0.5:    {best_f05:.4f}")
    print("="*60)
    
    # Save Model Artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/lgb_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "best_threshold": best_thresh,
            "features": FEATURE_NAMES
        }, f)
    print(f"Saved trained model and optimal threshold to {model_path}")

if __name__ == "__main__":
    train_and_optimize()
