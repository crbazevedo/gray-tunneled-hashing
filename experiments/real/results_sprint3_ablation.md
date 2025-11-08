# Sprint 3: Ablation Study Results

This document summarizes the ablation study comparing baseline methods vs. Gray-Tunneled Hashing variants.

## Overview

We systematically compared:
- **Baselines**: `baseline_sign`, `baseline_random_proj`
- **Gray-Tunneled variants**:
  - `gray_tunneled_trivial`: Simple mapping (identity/Gray-code), no optimization
  - `gray_tunneled_two_swap_only`: 2-swap hill climb only, no tunneling
  - `gray_tunneled_full`: 2-swap + tunneling (full optimization)

## Key Metrics

For each configuration, we measure:
- **Recall@k**: Fraction of true k nearest neighbors found
- **Build time**: Time to construct the index
- **Search time**: Time to search the index
- **QAP cost**: Final QAP objective value (for GT methods)

## Results Summary

### Method Comparison (Fixed n_bits=64, n_codes=512, k=10)

| Method | Recall@10 | Recall@50 | Recall@100 | Build Time (s) | Search Time (ms) |
|--------|-----------|-----------|------------|----------------|------------------|
| baseline_sign | TBD | TBD | TBD | TBD | TBD |
| baseline_random_proj | TBD | TBD | TBD | TBD | TBD |
| gray_tunneled_trivial | TBD | TBD | TBD | TBD | TBD |
| gray_tunneled_two_swap_only | TBD | TBD | TBD | TBD | TBD |
| gray_tunneled_full (random) | TBD | TBD | TBD | TBD | TBD |
| gray_tunneled_full (cluster) | TBD | TBD | TBD | TBD | TBD |

*Note: Results will be populated after running the sweep script.*

### Recall@k vs n_bits (Fixed n_codes=512, k=10)

| n_bits | baseline_sign | baseline_random_proj | gt_trivial | gt_two_swap | gt_full |
|--------|---------------|---------------------|-----------|-------------|---------|
| 32 | TBD | TBD | TBD | TBD | TBD |
| 64 | TBD | TBD | TBD | TBD | TBD |
| 128 | TBD | TBD | TBD | TBD | TBD |

### Recall@k vs n_codes (Fixed n_bits=64, k=10)

| n_codes | baseline_sign | baseline_random_proj | gt_trivial | gt_two_swap | gt_full |
|---------|---------------|---------------------|-----------|-------------|---------|
| 256 | TBD | TBD | TBD | TBD | TBD |
| 512 | TBD | TBD | TBD | TBD | TBD |
| 1024 | TBD | TBD | TBD | TBD | TBD |

## Observations

### When GT-full Wins

*Analysis will be added after sweep results are available.*

### When 2-swap-only is Sufficient

*Analysis will be added after sweep results are available.*

### When GT is Close to Baselines

*Analysis will be added after sweep results are available.*

## Block Strategy Comparison

### Random vs Cluster-Based Blocks

| Configuration | GT-full (random) | GT-full (cluster) | Difference |
|---------------|------------------|-------------------|------------|
| n_bits=64, n_codes=512, block_size=8 | TBD | TBD | TBD |
| n_bits=64, n_codes=512, block_size=16 | TBD | TBD | TBD |

**Preliminary Recommendation**: 
- *Will be determined after analysis*

## Build Time vs Quality Tradeoff

*Analysis of build time vs recall@k tradeoff will be added.*

## Next Steps

1. Run the sweep script: `python scripts/run_sweep_sprint3.py`
2. Analyze results from `results_sprint3_sweep.csv`
3. Update this document with actual numbers and insights

