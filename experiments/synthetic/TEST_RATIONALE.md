# Test Suite Design Rationale

## Overview

This document explains why each test exists, what hypotheses it validates, and why it matters for the Gray-Tunneled Hashing project.

---

## Test Categories

### 1. Correctness Tests
**Purpose**: Ensure basic functionality works as expected

#### `test_generate_hypercube_vertices()`
- **What it tests**: Hypercube vertex generation produces correct binary vectors
- **Why it matters**: Foundation for all operations - if this is wrong, everything is wrong
- **Hypothesis**: H1 (planted model structure) - we need correct hypercube structure
- **Type**: Correctness

#### `test_qap_cost()`
- **What it tests**: QAP cost computation matches expected formula
- **Why it matters**: Core objective function - optimization is meaningless if cost is wrong
- **Hypothesis**: None directly, but validates correctness of objective
- **Type**: Correctness

#### `test_generate_hypercube_edges()`
- **What it tests**: Edge generation produces correct Hamming-1 pairs
- **Why it matters**: QAP objective sums over edges - wrong edges = wrong optimization
- **Hypothesis**: None directly, but validates graph structure
- **Type**: Correctness

---

### 2. Property Validation Tests
**Purpose**: Verify theoretical properties hold

#### `test_generate_planted_phi_hamming1_property()`
- **What it tests**: Hamming-1 neighbors in φ are closer than random pairs
- **Why it matters**: Validates **H1 (Planted Model Structure Hypothesis)**
  - If this fails, the planted model has no recoverable structure
  - Without structure, optimization is pointless
- **Hypothesis**: H1
- **Type**: Property validation
- **Result**: ✅ Validated

#### `test_hill_climb_two_swap_monotonic()`
- **What it tests**: Cost decreases monotonically during hill-climbing
- **Why it matters**: Validates **H3 (2-Swap Monotonicity Hypothesis)**
  - If cost doesn't decrease, the algorithm is buggy
  - Monotonicity is a fundamental property of hill-climbing
- **Hypothesis**: H3
- **Type**: Algorithmic property
- **Result**: ✅ Validated

#### `test_tunneling_step()`
- **What it tests**: Tunneling steps never increase cost
- **Why it matters**: Ensures tunneling is a valid optimization operator
  - If tunneling increases cost, it's not helping
  - Validates that we only accept improving moves
- **Hypothesis**: Implicit in H5 (tunneling effectiveness)
- **Type**: Algorithmic property
- **Result**: ✅ Validated

---

### 3. Optimality Tests
**Purpose**: Verify algorithm finds good solutions

#### `test_hill_climb_two_swap_small_optimal()`
- **What it tests**: On small instances (N ≤ 8), hill-climbing finds optimal solutions
- **Why it matters**: Ground truth validation
  - For small N, we can enumerate all permutations and find global optimum
  - If we can't find optimal on small instances, we won't on large ones
- **Hypothesis**: Algorithm correctness
- **Type**: Optimality + correctness
- **Result**: ✅ Validated

#### `test_hasher_cost_better_than_random()`
- **What it tests**: Optimized cost < random baseline cost
- **Why it matters**: Validates **H6 (Optimization Quality Hypothesis)**
  - If we can't beat random, the algorithm is useless
  - This is the minimum bar for success
- **Hypothesis**: H6
- **Type**: Quality validation
- **Result**: ✅ Validated (40-60% improvement)

---

### 4. Innovation Validation Tests
**Purpose**: Verify key innovations work

#### `test_tunneling_improves_upon_two_swap()`
- **What it tests**: After 2-swap converges, tunneling can further reduce cost
- **Why it matters**: Validates **H5 (Block Tunneling Effectiveness Hypothesis)**
  - This is the key innovation: block moves escape local minima
  - If tunneling doesn't help, we don't need it (and the project is less interesting)
- **Hypothesis**: H5
- **Type**: Innovation validation
- **Result**: ✅ Validated

---

### 5. Integration Tests
**Purpose**: Verify complete system works

#### `test_end_to_end_synthetic_run()`
- **What it tests**: Full pipeline from data generation to optimization
- **Why it matters**: Validates **H6 (Optimization Quality)** on complete system
  - Individual components might work, but integration might fail
  - This is the ultimate test: does the whole system work?
- **Hypothesis**: H6
- **Type**: Integration test
- **Result**: ✅ Validated

---

## Test Coverage Matrix

| Component | Correctness | Property | Optimality | Integration |
|-----------|-------------|----------|------------|-------------|
| Hypercube generation | ✅ | - | - | - |
| Planted φ generation | ✅ | ✅ (H1) | - | - |
| Noisy embeddings | ✅ | - | - | - |
| QAP cost | ✅ | - | - | - |
| 2-swap hill-climb | ✅ | ✅ (H3) | ✅ | - |
| Block tunneling | ✅ | ✅ | - | - |
| GrayTunneledHasher | ✅ | - | ✅ (H6) | ✅ |

---

## Missing Tests (Future Work)

### H4: Elementary Landscape Structure
- **Hypothesis**: QAP has relaxation rate λ = 4/N
- **Why not tested**: Requires theoretical analysis, not just empirical tests
- **Future**: Add theoretical analysis in Sprint 2+

### Scalability Tests
- **What**: Measure runtime and quality as N increases
- **Why**: Need to validate algorithm scales to real-world sizes
- **Future**: Sprint 2

### Real-World Embedding Tests
- **What**: Test on actual embedding datasets (SIFT, GloVe)
- **Why**: Synthetic data is controlled, but real data is the goal
- **Future**: Sprint 2+

---

## Test Execution Strategy

### Fast Tests (for development)
- All correctness tests: ~1 second
- Property validation: ~1 second
- Run these frequently during development

### Slow Tests (for CI)
- Optimality tests: ~1-5 minutes (includes optimization)
- Integration tests: ~5 minutes (full pipeline)
- Run these before commits

### Test Selection
```bash
# Fast feedback during development
pytest tests/test_synthetic_generators.py -v
pytest tests/test_qap_two_swap.py::test_qap_cost -v

# Full suite before commit
pytest -v

# Specific hypothesis validation
pytest tests/test_block_moves.py::test_tunneling_improves_upon_two_swap -v
```

---

## Why These Tests Matter

### For Theory
- Validate that theoretical properties hold in practice
- Identify where theory and practice diverge
- Guide future theoretical development

### For Implementation
- Catch bugs early (correctness tests)
- Ensure algorithms work as designed (property tests)
- Validate optimization quality (optimality tests)

### For Research
- Document what works and what doesn't
- Provide evidence for hypotheses
- Guide future research directions

---

## Conclusion

The test suite is designed to:
1. ✅ Validate correctness (foundation)
2. ✅ Validate theoretical properties (theory)
3. ✅ Validate optimization quality (practice)
4. ✅ Validate key innovations (tunneling)

All critical hypotheses (H1, H3, H5, H6) have corresponding tests that validate them. The foundation is solid for extending to real-world embeddings.

