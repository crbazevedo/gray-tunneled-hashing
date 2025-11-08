# Empirical Landscape Analysis - Sprint 3

This document connects empirical observations from Sprint 3 experiments to the theoretical foundations of Gray-Tunneled Hashing.

## Overview

We analyze:
1. Distribution of 2-swap local minima
2. Effectiveness of tunneling in escaping local minima
3. Patterns vs hyperparameters (n_bits, n_codes, block_size)

## 2-Swap Local Minima Distribution

### Empirical Observations

From `analyze_two_swap_landscape.py`:

- **Distribution shape**: *To be filled after running experiments*
- **Spread**: Standard deviation of final costs indicates landscape ruggedness
- **Multiple basins**: Number of distinct local minima found

### Connection to Elementary Landscape Theory

The QAP objective under 2-swap moves forms an **elementary landscape**:
- Each local minimum is reachable via a sequence of improving 2-swaps
- The number of local minima grows with problem size
- Empirical distribution helps validate theoretical predictions

### Key Questions

1. How many distinct local minima exist?
2. What is the typical cost gap between best and worst local minima?
3. How does this vary with n_bits and n_codes?

## Tunneling Effectiveness

### Empirical Observations

From `analyze_tunneling_effect_sprint3.py`:

- **Improvement rate**: Fraction of local minima that tunneling improves
- **Typical improvement magnitude**: Mean/std of cost reductions
- **Block size effects**: How block_size affects tunneling success

### Connection to Block Tunneling Theory

Tunneling moves allow escaping local minima by:
- Reoptimizing assignments within a block
- Exploiting the structure of the hypercube graph
- Potentially finding better global minima

### Key Questions

1. What fraction of local minima can tunneling improve?
2. How much improvement is typical?
3. Does cluster-based block selection outperform random?

## Patterns vs Hyperparameters

### n_bits Effects

- **Larger n_bits**: More hypercube vertices, potentially more local minima
- **Expected**: More opportunities for optimization, but also harder search

### n_codes Effects

- **Larger n_codes**: More codebook vectors, richer representation
- **Expected**: Better recall, but higher optimization cost

### block_size Effects

- **Smaller blocks**: Faster reoptimization, but limited scope
- **Larger blocks**: More global improvements, but exponential cost

## Comparison with Planted Model Predictions

From Sprint 1 synthetic experiments, we observed:
- Clear benefits of Gray-Tunneled on planted instances
- Tunneling consistently improved 2-swap local minima

For real embeddings:
- *Do we see similar patterns?*
- *Are the improvements as consistent?*
- *What explains differences?*

## Future Directions

1. **Theoretical analysis**: Formal bounds on number of local minima
2. **Adaptive strategies**: Dynamic block_size based on landscape
3. **Hybrid methods**: Combine multiple strategies based on empirical patterns

## References

- Elementary landscape theory: [Stadler 1996]
- QAP optimization: [Burkard et al. 2009]
- Block tunneling: Gray-Tunneled Hashing theory document

