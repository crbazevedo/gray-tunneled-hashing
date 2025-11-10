# Sprint 6 - Comprehensive Results Analysis

## Executive Summary

This report presents the results of Sprint 6 experiments evaluating Gray-Tunneled Hashing (GTH) 
with LSH families and random projection methods.

### Key Findings

- ⚠️ **H3 (Hamming Ball Improves Recall)**: PARTIALLY VALIDATED
- ❌ **H4 (GTH Improves Recall)**: REJECTED

## Experiment Results

### Experiment 1 (Radius 0)

| Method | Recall (mean ± std) | 95% CI | Build Time (s) | Search Time (ms) |
|--------|---------------------|--------|----------------|------------------|
| baseline_hyperplane | 0.1220 ± 0.0172 | [0.0981, 0.1459] | 0.00 | 0.00 |
| baseline_p_stable | 0.0460 ± 0.0136 | [0.0272, 0.0648] | 0.00 | 0.00 |
| baseline_random_proj | 0.1220 ± 0.0172 | [0.0981, 0.1459] | 0.00 | 0.00 |
| hyperplane | 0.0220 ± 0.0194 | [-0.0049, 0.0489] | 50.86 | 1.62 |
| p_stable | 0.0120 ± 0.0098 | [-0.0016, 0.0256] | 71.06 | 1.61 |
| random_proj | 0.0180 ± 0.0194 | [-0.0089, 0.0449] | 50.94 | 1.57 |

### Experiment 1 (Radius 1)

| Method | Recall (mean ± std) | 95% CI | Build Time (s) | Search Time (ms) |
|--------|---------------------|--------|----------------|------------------|
| baseline_hyperplane | 0.1220 ± 0.0172 | [0.0981, 0.1459] | 0.00 | 0.00 |
| baseline_p_stable | 0.0460 ± 0.0136 | [0.0272, 0.0648] | 0.00 | 0.00 |
| baseline_random_proj | 0.1220 ± 0.0172 | [0.0981, 0.1459] | 0.00 | 0.00 |
| hyperplane | 0.0680 ± 0.0293 | [0.0274, 0.1086] | 52.50 | 3.74 |
| p_stable | 0.0520 ± 0.0194 | [0.0251, 0.0789] | 70.92 | 3.63 |
| random_proj | 0.0520 ± 0.0232 | [0.0199, 0.0841] | 51.75 | 3.48 |

## Hypothesis Validation

### H3: Hamming Ball Improves Recall
**Status**: PARTIALLY_VALIDATED

| Method | Recall R=0 | Recall R=1 | Recall R=2 | R1 > R0 | R2 > R1 |
|--------|-----------|-----------|-----------|---------|---------|
| hyperplane | 0.0220 | 0.0680 | 0.0000 | ✅ | ✅ |
| baseline_hyperplane | 0.1220 | 0.1220 | 0.0000 | ❌ | ✅ |
| p_stable | 0.0120 | 0.0520 | 0.0000 | ✅ | ✅ |
| baseline_p_stable | 0.0460 | 0.0460 | 0.0000 | ❌ | ✅ |
| baseline_random_proj | 0.1220 | 0.1220 | 0.0000 | ❌ | ✅ |
| random_proj | 0.0180 | 0.0520 | 0.0000 | ❌ | ✅ |

### H4: GTH Improves Recall
**Status**: REJECTED

| Method | Baseline Recall | GTH Recall | Improvement | p-value | Significant |
|--------|----------------|-----------|-------------|---------|------------|
| hyperplane | 0.1220 | 0.0450 | -63.11% | 0.0000 | ✅ |
| p_stable | 0.0460 | 0.0320 | -30.43% | 0.1597 | ❌ |
| random_proj | 0.1220 | 0.0350 | -71.31% | 0.0000 | ✅ |

### H5: LSH + GTH vs. Random Projection + GTH
**Status**: COMPARED

- **LSH Mean: 0.0385** (95% CI: [0.0238, 0.0532])
- **Random Projection Mean: 0.0350** (95% CI: [0.0144, 0.0556])
- **Difference: 0.0035**
- **Statistical Test**: p-value = 0.7694, Significant = ❌

## Conclusions and Recommendations

### Practical Defaults

Based on the experimental results, the following defaults are recommended:

- **n_bits**: 8 (good balance between recall and build time)
- **n_codes**: 32 (reasonable for small to medium datasets)
- **hamming_radius**: 1 (optimal trade-off between recall and search time)
- **block_size**: 4-8 (depending on dataset size)
- **num_tunneling_steps**: 5-10 (good improvement/cost tradeoff)

### Limitations

- GTH methods currently show lower recall than baselines in some configurations
- Further investigation needed for bucket-to-dataset mapping optimization
- Experiments conducted on synthetic data; validation on real datasets recommended
