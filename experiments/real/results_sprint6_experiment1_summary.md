# Sprint 6 - Experiment 1: Hamming Ball Radius Validation

## Configuration

- n_bits: 6
- n_codes: 16
- k: 5
- n_samples: 100
- n_queries: 20
- n_runs: 5
- hamming_radius: 0, 1, 2

## Results Summary

### Radius 0 (Exact Match)

| Method | Recall (mean ± std) |
|--------|---------------------|
| baseline_hyperplane | 0.1220 ± 0.0172 |
| baseline_p_stable | 0.0460 ± 0.0136 |
| baseline_random_proj | 0.1220 ± 0.0172 |
| hyperplane (GTH) | 0.0220 ± 0.0194 |
| p_stable (GTH) | 0.0120 ± 0.0098 |
| random_proj (GTH) | 0.0180 ± 0.0194 |

### Radius 1

| Method | Recall (mean ± std) |
|--------|---------------------|
| baseline_hyperplane | 0.1220 ± 0.0172 |
| baseline_p_stable | 0.0460 ± 0.0136 |
| baseline_random_proj | 0.1220 ± 0.0172 |
| hyperplane (GTH) | 0.0680 ± 0.0293 |
| p_stable (GTH) | 0.0520 ± 0.0194 |
| random_proj (GTH) | 0.0520 ± 0.0232 |

### Radius 2

| Method | Recall (mean ± std) |
|--------|---------------------|
| baseline_hyperplane | 0.1220 ± 0.0172 |
| baseline_p_stable | 0.0460 ± 0.0136 |
| baseline_random_proj | 0.1220 ± 0.0172 |
| hyperplane (GTH) | 0.0520 ± 0.0075 |
| p_stable (GTH) | 0.0560 ± 0.0185 |
| random_proj (GTH) | 0.0500 ± 0.0261 |

## Observations

1. **Hamming Ball Expansion Helps**: Recall improves from radius 0 to radius 1:
   - hyperplane: 0.022 → 0.068 (3.1x improvement)
   - p_stable: 0.012 → 0.052 (4.3x improvement)
   - random_proj: 0.018 → 0.052 (2.9x improvement)

2. **Radius 2 Not Better**: Recall at radius 2 is similar or slightly worse than radius 1:
   - This suggests diminishing returns or potential over-expansion

3. **GTH Still Below Baseline**: All GTH methods have lower recall than baselines:
   - This indicates a potential issue with the bucket → dataset mapping or the permutation application
   - Requires further investigation

4. **Build Time**: GTH methods take ~50-70s to build, while baselines are instant

## Hypothesis Validation

- **H3 (Hamming Ball Improves Recall)**: ✅ **PARTIALLY VALIDATED**
  - Recall increases from radius 0 to radius 1
  - However, radius 2 does not improve further
  - GTH methods still below baseline

## Next Steps

1. Investigate why GTH recall is below baseline
2. Test with larger n_bits and n_codes
3. Validate bucket → dataset mapping more thoroughly
4. Execute Experiment 2 (LSH vs. Random Projection comparison)

