# Synthetic Experiments

This directory contains configurations, notebooks, and results for synthetic experiments with Gray-Tunneled Hashing.

Synthetic experiments are used to:
- Validate the algorithm on controlled data
- Test theoretical properties
- Benchmark performance on different data distributions
- Compare against baseline hashing methods

## Running Synthetic Experiments

### Basic Usage

From the repository root, run:

```bash
python scripts/run_synthetic_experiment.py --n-bits 5 --dim 8 --sigma 0.1
```

### Command-Line Options

- `--n-bits`: Number of bits (hypercube dimension). Determines N = 2^n_bits. Recommended: 4-8 for quick tests, up to 10 for thorough experiments.
- `--dim`: Embedding dimension (default: 8)
- `--sigma`: Noise standard deviation in planted model (default: 0.1)
- `--block-size`: Block size for tunneling moves (default: 8)
- `--max-two-swap-iters`: Maximum iterations for 2-swap hill climbing (default: 50)
- `--num-tunneling-steps`: Number of tunneling steps to perform (default: 10)
- `--random-state`: Random seed for reproducibility (default: 42)

### Example Runs

**Quick test (small instance):**
```bash
python scripts/run_synthetic_experiment.py --n-bits 4 --dim 8 --sigma 0.1 --max-two-swap-iters 20 --num-tunneling-steps 5
```

**Medium instance:**
```bash
python scripts/run_synthetic_experiment.py --n-bits 5 --dim 16 --sigma 0.1 --max-two-swap-iters 50 --num-tunneling-steps 10
```

**Large instance (takes longer):**
```bash
python scripts/run_synthetic_experiment.py --n-bits 6 --dim 32 --sigma 0.1 --max-two-swap-iters 100 --num-tunneling-steps 20
```

## Experiment Output

The script prints:
- Parameters used
- Step-by-step progress
- Results summary including:
  - Costs (random baseline, identity baseline, Gray-Tunneled)
  - Relative improvements
  - Distance to planted π* (ground truth)
  - Cost history during optimization

## Results

See `results_sprint1.md` for initial numerical results from Sprint 1.

## Experiment Types

### Distance Preservation
Tests how well the algorithm preserves pairwise distances between embeddings when mapped to binary codes.

### Scalability
Tests performance with varying dataset sizes (n_bits) and embedding dimensions.

### Code Length Analysis
Tests the effect of different binary code lengths (n_bits) on optimization quality.

### Noise Robustness
Tests how the algorithm performs under different noise levels (sigma parameter).
