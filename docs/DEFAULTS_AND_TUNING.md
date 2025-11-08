# Defaults and Tuning Guide

This document provides practical defaults and tuning guidelines for Gray-Tunneled Hashing based on Sprint 3 empirical results.

## Quick-Start Recommendations

### If Your Dataset is Like X and You Care About Y

| Dataset Type | Priority | Recommended Settings |
|--------------|----------|---------------------|
| Text embeddings (small, <100K) | Quality | `n_bits=64`, `n_codes=512`, `mode="full"`, `block_size=8`, `num_tunneling_steps=10` |
| Text embeddings (medium, 100K-1M) | Quality | `n_bits=128`, `n_codes=1024`, `mode="full"`, `block_size=8`, `num_tunneling_steps=10` |
| Text embeddings (large, >1M) | Latency | `n_bits=64`, `n_codes=256`, `mode="two_swap_only"`, `block_size=8` |
| Image embeddings | Quality | `n_bits=128`, `n_codes=1024`, `mode="full"`, `block_size=16`, `num_tunneling_steps=15` |
| Any (build time constrained) | Speed | `n_bits=32`, `n_codes=256`, `mode="two_swap_only"` |

*Note: These are preliminary recommendations. Update after Sprint 3 sweep results.*

## Default Parameters

Based on Sprint 3 analysis:

```python
DEFAULT_N_BITS = 64  # Good balance for most text embeddings
DEFAULT_N_CODES = 512  # Reasonable for datasets <1M
DEFAULT_BLOCK_SIZE = 8  # Optimal for most cases
DEFAULT_NUM_TUNNELING_STEPS = 10  # Good improvement/cost tradeoff
DEFAULT_MODE = "full"  # Best quality, but slower
DEFAULT_BLOCK_SELECTION_STRATEGY = "random"  # Or "cluster" if codebook available
```

## Tuning Guidelines

### When to Disable Tunneling

**Use `mode="two_swap_only"` when:**
- Build time is critical (< 1 minute required)
- Dataset is very large (> 10M vectors)
- Quality gains from tunneling are marginal (< 2% recall improvement)

**Use `mode="trivial"` when:**
- Extremely fast build time needed (< 10 seconds)
- Quality is acceptable even without optimization
- Prototyping or initial exploration

### When to Increase n_bits vs n_codes

**Increase `n_bits` when:**
- You need higher recall@k (more bits = more hypercube vertices = better discrimination)
- Dataset has high intrinsic dimensionality
- You can afford slightly longer build times

**Increase `n_codes` when:**
- Dataset has diverse clusters (need more codebook vectors)
- You want better approximation of the embedding distribution
- You can afford longer codebook build time (k-means scales with n_codes)

**Rule of thumb:**
- `n_bits` primarily affects search quality (more bits = better recall)
- `n_codes` primarily affects representation quality (more codes = better approximation)

### Block Size Selection

**Smaller `block_size` (4-8):**
- Faster tunneling steps
- More frequent improvements
- Good for large datasets

**Larger `block_size` (12-16):**
- Potentially better global improvements
- Slower per step, but fewer steps needed
- Good for smaller datasets or when quality is critical

**Default: `block_size=8`** - Good balance for most cases

### Block Selection Strategy

**Random blocks:**
- Faster selection
- Works well in most cases
- Default recommendation

**Cluster-based blocks:**
- Potentially better improvements (exploits structure)
- Requires cluster assignments from codebook
- May be slower to select blocks
- Use when codebook is available and quality is critical

### Number of Tunneling Steps

**Fewer steps (5-10):**
- Faster build time
- Good for large datasets
- Most improvements happen in first few steps

**More steps (15-20):**
- Better chance of finding global optimum
- Diminishing returns after ~10 steps
- Use when quality is critical and build time is acceptable

**Default: `num_tunneling_steps=10`** - Good improvement/cost tradeoff

## Tuning Workflow

1. **Start with defaults**: Use recommended defaults for your dataset type
2. **Run quick experiment**: Test on a sample of your data
3. **Measure recall@k**: Is it acceptable?
4. **If too low**:
   - Increase `n_bits` first (better recall)
   - If still low, increase `n_codes` (better representation)
   - Consider enabling tunneling (`mode="full"`)
5. **If build time too high**:
   - Reduce `num_tunneling_steps`
   - Try `mode="two_swap_only"`
   - Reduce `n_codes`
6. **If search latency too high**:
   - Reduce `n_bits` (smaller codes = faster search)
   - Consider using FAISS backend
7. **Iterate**: Fine-tune based on your specific requirements

## Example Configurations

### High Quality (slow build OK)
```python
n_bits = 128
n_codes = 1024
mode = "full"
block_size = 8
num_tunneling_steps = 15
block_selection_strategy = "cluster"
```

### Balanced (default)
```python
n_bits = 64
n_codes = 512
mode = "full"
block_size = 8
num_tunneling_steps = 10
block_selection_strategy = "random"
```

### Fast Build (quality OK)
```python
n_bits = 64
n_codes = 256
mode = "two_swap_only"
block_size = 8
```

### Very Fast (prototype)
```python
n_bits = 32
n_codes = 256
mode = "trivial"
```

## Performance Characteristics

### Build Time Scaling

- **Codebook (k-means)**: O(n_base * n_codes * dim * iterations)
- **2-swap optimization**: O(n_codes^2 * max_iter * sample_size)
- **Tunneling**: O(num_tunneling_steps * block_size! * num_blocks)

### Search Time Scaling

- **Hamming search**: O(n_base * n_bits) for Python, O(n_base * log(n_base)) for FAISS
- **Code lookup**: O(1) per query

### Quality Scaling

- **Recall@k**: Generally increases with n_bits and n_codes
- **Diminishing returns**: After ~128 bits, improvements are small

## Troubleshooting

### Low Recall@k

- Increase `n_bits`
- Increase `n_codes`
- Enable tunneling (`mode="full"`)
- Check if ground truth is correct

### High Build Time

- Reduce `num_tunneling_steps`
- Use `mode="two_swap_only"`
- Reduce `n_codes`
- Reduce `max_two_swap_iters`

### High Search Latency

- Reduce `n_bits`
- Use FAISS backend (if available)
- Consider approximate search strategies

## References

- Sprint 3 sweep results: `experiments/real/results_sprint3_sweep.csv`
- Ablation analysis: `experiments/real/results_sprint3_ablation.md`
- Landscape analysis: `theory/empirical_landscape_notes_sprint3.md`

