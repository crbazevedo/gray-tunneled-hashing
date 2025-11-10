# Sprint 5 Results

## Overview

This document summarizes results from Sprint 5 experiments comparing LSH families vs. random projection with GTH.

## Experimental Setup

### Configuration

- **n_bits**: 6, 8 (tested)
- **n_codes**: 16, 32 (tested)
- **n_samples**: 50-100
- **n_queries**: 10-20
- **k**: 3-5
- **hamming_radius**: 0, 1, 2
- **n_runs**: 1-3 (for averaging)

### Methods Compared

1. **Baseline Methods** (without GTH):
   - `baseline_hyperplane`: Hyperplane LSH only
   - `baseline_p_stable`: p-stable LSH only
   - `baseline_random_proj`: Random projection only

2. **GTH Methods** (with GTH optimization):
   - `hyperplane`: Hyperplane LSH + GTH
   - `p_stable`: p-stable LSH + GTH
   - `random_proj`: Random projection + GTH

## Results Summary

### Experiment 1: Basic Validation (n_bits=6, n_codes=16, radius=1)

| Method | Recall (mean ± std) | Build Time (s) | Search Time (ms) |
|--------|---------------------|----------------|------------------|
| baseline_hyperplane | 0.2000 ± 0.0000 | 0.00 | 0.0 |
| baseline_p_stable | 0.1333 ± 0.0000 | 0.00 | 0.0 |
| baseline_random_proj | 0.2000 ± 0.0000 | 0.00 | 0.0 |
| hyperplane | 0.0667 ± 0.0000 | 11.74 | N/A |
| p_stable | 0.0000 ± 0.0000 | 26.53 | N/A |
| random_proj | 0.0000 ± 0.0000 | 18.57 | N/A |

**Note**: Initial results show lower recall for GTH methods. This may indicate:
1. Issues with bucket-to-dataset mapping
2. Need for better ground truth generation
3. Small dataset size affecting results

## Hypothesis Validation

### H2: GTH Preserves Collisions LSH

**Status**: ✅ **VALIDATED**

- **Métrica**: Collision Preservation Rate
- **Resultado**: 100% preservation (validated in tests)
- **Evidência**: All 3 collision validation tests pass

### H3: Hamming Ball Improves Recall

**Status**: ⚠️ **PENDING VALIDATION**

- **Métrica**: recall@k para radius=0, 1, 2
- **Esperado**: recall(radius=2) > recall(radius=1) > recall(radius=0)
- **Status**: Requires experiments with multiple radius values

### H4: GTH Improves Recall

**Status**: ⚠️ **PENDING VALIDATION**

- **Métrica**: Improvement Over Baseline
- **Esperado**: recall_gth > recall_baseline
- **Status**: Initial results show opposite trend - requires investigation

### H5: LSH + GTH vs. Random Projection + GTH

**Status**: ⚠️ **PENDING VALIDATION**

- **Métrica**: recall@k comparativo
- **Status**: Requires more experiments

## Observations

1. **Collision Preservation**: ✅ Confirmed 100% preservation in all tests
2. **Build Time**: GTH methods have significant build time overhead (10-30s vs. <1s)
3. **Recall Performance**: Initial results show baseline methods outperforming GTH - requires investigation

## Limitations

1. **Small Dataset**: Initial experiments use small datasets (50-100 samples)
2. **Ground Truth**: Ground truth generation may need refinement
3. **Bucket Mapping**: Bucket-to-dataset index mapping may have issues

## Next Steps

1. Investigate why GTH methods show lower recall
2. Run experiments with multiple radius values (0, 1, 2)
3. Validate bucket-to-dataset mapping
4. Run larger-scale experiments
5. Compare with theoretical expectations

## Files

- Results JSON: `experiments/real/results_sprint5_experiment1.json`
- Analysis script: `scripts/analyze_sprint5_results.py`
- Benchmark script: `scripts/benchmark_lsh_vs_random_proj.py`

