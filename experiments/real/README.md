# Real Embeddings Experiments

This directory contains experiments and results for Gray-Tunneled Hashing on real-world embeddings.

## Dataset

We use text embeddings from a real dataset (e.g., Quora Question Pairs or similar).

### Data Format

Embeddings are stored in `.npy` format:
- `{dataset}_base_embeddings.npy`: Base corpus embeddings (N, dim)
- `{dataset}_queries_embeddings.npy`: Query embeddings (Q, dim)
- `{dataset}_ground_truth_indices.npy`: Ground truth kNN indices (Q, k)

### Generating Ground Truth

Before running experiments, compute ground truth:

```bash
python scripts/compute_float_ground_truth.py \
    --dataset quora \
    --k 100 \
    --method auto
```

This computes exact kNN using float embeddings (FAISS or brute-force) and saves ground truth indices.

## Running Experiments

### Baseline Binary Experiments

Run baseline binary experiments (sign or random projection):

```bash
# Sign thresholding
python scripts/run_real_experiment_baseline.py \
    --dataset quora \
    --method sign \
    --k 10

# Random projection
python scripts/run_real_experiment_baseline.py \
    --dataset quora \
    --method random_proj \
    --n-bits 64 \
    --k 10 \
    --random-state 42
```

### Gray-Tunneled Experiments

Run Gray-Tunneled Hashing experiments:

```bash
python scripts/run_real_experiment_gray_tunneled.py \
    --dataset quora \
    --n-bits 64 \
    --n-codes 256 \
    --k 10 \
    --block-size 8 \
    --max-two-swap-iters 50 \
    --num-tunneling-steps 10 \
    --random-state 42
```

### Parameters

**Baseline parameters**:
- `--method`: `sign` or `random_proj`
- `--n-bits`: Number of bits (only for `random_proj`, default: 64)
- `--k`: Recall@k value (default: 10)

**Gray-Tunneled parameters**:
- `--n-bits`: Number of bits for binary codes (default: 64)
- `--n-codes`: Number of codebook vectors (must be ≤ 2**n_bits)
- `--k`: Recall@k value (default: 10)
- `--block-size`: Block size for tunneling (default: 8)
- `--max-two-swap-iters`: Max iterations for 2-swap (default: 50)
- `--num-tunneling-steps`: Number of tunneling steps (default: 10)

## Results

Results are saved as JSON files in this directory:
- `results_baseline_{method}_{dataset}_k{k}.json`: Baseline results
- `results_gray_tunneled_{dataset}_bits{n_bits}_codes{n_codes}_k{k}.json`: Gray-Tunneled results

Each result file contains:
- `recall_at_k`: Recall@k score (0.0 to 1.0)
- `binarize_time` / `codebook_time` / `optimization_time`: Timing information
- `build_time`: Index build time
- `search_time`: Total search time
- `avg_search_time_ms`: Average search time per query
- `backend`: FAISS or Python

See `results_sprint2.md` for detailed analysis and comparisons.

## FAISS Backend

FAISS is optional but recommended for better performance. Install with:

```bash
pip install faiss-cpu
# or for GPU:
pip install faiss-gpu
```

If FAISS is not available, the code automatically falls back to a pure Python implementation.

## Directory Structure

```
experiments/real/
├── README.md                    # This file
├── results_sprint2.md          # Detailed results and analysis
└── data/                       # Embeddings (gitignored)
    ├── {dataset}_base_embeddings.npy
    ├── {dataset}_queries_embeddings.npy
    └── {dataset}_ground_truth_indices.npy
```

