# Business Entity Resolution Pipeline

## Overview
This package contains the complete, self-contained end-to-end pipeline for the ML Challenge 2026 Business Entity Resolution challenge.
The system implements a high-performance 2-stage architecture:
1. **Multi-Channel Vectorized Blocking** (Country-partitioned hash equi-joins in DuckDB with Hindi transliteration and French accent stripping).
2. **Precision-Weighted Pairwise Classification** (RapidFuzz string similarity features + LightGBM classifier optimized for Macro F0.5).

## Directory Structure
```
business_entity_resolution/
├── src/
│   ├── cleaner.py                  # Text normalization, legal suffix cleaning, Hindi transliteration
│   ├── feature_extractor.py        # Pairwise lexical and address similarity metrics
│   ├── train_model.py              # LightGBM model training & F0.5 threshold tuning
│   └── generate_full_submission.py # Production candidate & match generation
├── requirements.txt                # Pinned dependencies
└── README.md                       # Reproduction guide
```

## Setup & Reproduction Instructions

### 1. Environment Installation
Ensure Python 3.9+ is installed, then install requirements:
```bash
pip install -r requirements.txt
```

### 2. Model Training (Optional, Pre-trained Artifact Available)
To re-train the LightGBM classifier and find optimal validation threshold:
```bash
python3 src/train_model.py
```

### 3. Generate Submission Outputs
To run the full end-to-end candidate blocking and matching inference on the Test dataset:
```bash
python3 src/generate_full_submission.py
```
This generates:
- `output/matching_results.tsv` (Final predictions scored on the leaderboard)
- `output/candidate_pairs.tsv` (Candidate blocking set)

### 4. Validate Submission
Run the official validator from the root directory:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
You should see: `PASS — no blocking issues found. Safe to submit.`
