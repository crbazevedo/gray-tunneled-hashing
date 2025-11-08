# Root Cause Analysis: J(φ*) > J(φ₀) Violation

## Problem Statement

The theoretical guarantee **J(φ*) ≤ J(φ₀)** is being violated, where:
- J(φ₀) is the distribution-aware cost of the original (unoptimized) layout
- J(φ*) is the distribution-aware cost of the optimized layout

## Root Cause Identified

**The QAP objective and J(φ) objective are fundamentally different:**

1. **QAP Cost**: `f(π) = Σ_{(u,v) ∈ edges} D_weighted[π(u), π(v)]`
   - Sums only over **hypercube edges** (Hamming-1 neighbors)
   - ~N·n_bits terms

2. **J(φ) Objective**: `J(φ) = Σ_{i,j} π_i · w_ij · d_H(φ(c_i), φ(c_j))`
   - Sums over **all bucket pairs** (i, j)
   - K² terms

### Evidence

Validation script `validate_qap_vs_j_phi.py` shows:
- **Correlation between QAP cost and J(φ): -0.45** (negative!)
- Minimizing QAP cost does **NOT** minimize J(φ)
- In fact, there appears to be an inverse relationship

### Why This Happens

1. **Different Summation Domains**:
   - QAP: Only hypercube edges (sparse)
   - J(φ): All bucket pairs (dense)

2. **Different Optimization Landscapes**:
   - A permutation that minimizes QAP cost may increase J(φ)
   - The objectives are not aligned

3. **Padding Issues**:
   - When K < 2**n_bits, we pad D_weighted
   - This padding doesn't reflect the true J(φ) objective

## Solutions

### Option 1: Direct J(φ) Optimization (Recommended)

Implement a hill-climbing algorithm that directly optimizes J(φ):

```python
def hill_climb_j_phi(permutation, pi, w, bucket_to_code, n_bits, ...):
    # Directly minimize J(φ) using 2-swap moves
    # Evaluate J(φ) cost for each candidate swap
    # Accept improving swaps
```

**Pros**:
- Guarantees J(φ*) ≤ J(φ₀) (by construction)
- Directly optimizes the metric we care about

**Cons**:
- More expensive (O(K²) per evaluation vs O(E) for QAP)
- Need to implement custom optimization

### Option 2: Approximate J(φ) via Weighted QAP

Modify D_weighted so that minimizing QAP approximates minimizing J(φ):

- Use a different edge set (not just Hamming-1)
- Weight edges to approximate all-pairs sum

**Pros**:
- Reuses existing QAP infrastructure

**Cons**:
- Approximation may not be exact
- Still no theoretical guarantee

### Option 3: Hybrid Approach

1. Use QAP for initial optimization (fast)
2. Refine with direct J(φ) optimization (accurate)

## Implementation Status

- ✅ Root cause identified
- ✅ Validation script created
- ⏳ Direct J(φ) optimization (in progress)
- ⏳ Integration with GrayTunneledHasher (pending)

## Next Steps

1. Implement `hill_climb_j_phi` in `j_phi_objective.py`
2. Integrate with `GrayTunneledHasher.fit_with_traffic()`
3. Update `benchmark_distribution_aware_theoretical.py` to use direct optimization
4. Validate guarantee holds with new implementation

## Files Modified

- `scripts/validate_qap_vs_j_phi.py`: Validation script showing correlation
- `src/gray_tunneled_hashing/distribution/j_phi_objective.py`: Direct J(φ) optimization (new)
- `experiments/real/ROOT_CAUSE_ANALYSIS.md`: This document

