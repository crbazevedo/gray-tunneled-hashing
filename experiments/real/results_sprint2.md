# Sprint 2: Real Embeddings Results

## Overview

This document summarizes results from Sprint 2 experiments comparing baseline binary methods with Gray-Tunneled Hashing on real-world embeddings.

## Experimental Setup

### Dataset

- **Dataset**: TBD (Quora Question Pairs or similar)
- **Base embeddings**: Shape (N, dim)
- **Query embeddings**: Shape (Q, dim)
- **Ground truth**: Exact kNN computed via float embeddings

### Methods Compared

1. **Baseline - Sign**: Sign thresholding binarization
2. **Baseline - Random Projection**: Random projection binarization
3. **Gray-Tunneled**: Codebook k-means + Gray-Tunneled optimization

### Evaluation Metric

- **Recall@k**: Fraction of ground truth neighbors found in top-k retrieved results

## Results

### Example Results Table

| Method | n_bits | n_codes | k | Recall@k | Search Time (ms) | Notes |
|--------|--------|---------|---|----------|------------------|-------|
| Sign | - | - | 10 | TBD | TBD | Uses all dimensions |
| Random Proj | 64 | - | 10 | TBD | TBD | Baseline |
| Random Proj | 128 | - | 10 | TBD | TBD | More bits |
| Gray-Tunneled | 64 | 256 | 10 | TBD | TBD | Small codebook |
| Gray-Tunneled | 64 | 512 | 10 | TBD | TBD | Medium codebook |
| Gray-Tunneled | 64 | 1024 | 10 | TBD | TBD | Large codebook |

### Observations

**When Gray-Tunneled wins:**
- TBD (to be filled with actual results)

**When Gray-Tunneled loses:**
- TBD (to be filled with actual results)

**Hypotheses to validate:**
- **H7**: Gray-Tunneled improves recall@k vs baseline
- **H8**: Codebook size impacts performance (optimal n_codes exists)
- **H9**: Tunneling helps in real embeddings

## Analysis

### Recall@k vs k

How does recall change as k increases?

### Recall@k vs n_bits

How does recall change with more bits?

### Recall@k vs n_codes

For Gray-Tunneled, how does codebook size impact recall?

### Timing Analysis

- **Binarization time**: Baseline vs Gray-Tunneled codebook construction
- **Optimization time**: Gray-Tunneled QAP optimization overhead
- **Search time**: Query latency comparison

## Future Work

Based on results, prioritize:
- [ ] Tuning n_codes for optimal recall
- [ ] Testing different block sizes for tunneling
- [ ] Comparing with/without tunneling steps
- [ ] Scaling to larger datasets
- [ ] Testing different embedding dimensions

## Notes

This is a placeholder document. Actual results will be added after running experiments on real embeddings.

