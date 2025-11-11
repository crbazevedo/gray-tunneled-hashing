# Gray-Tunneled Hashing (GTH)

A novel binary hashing approach that optimizes the assignment of embeddings to binary codes by solving a Quadratic Assignment Problem (QAP), enabling efficient approximate nearest neighbor search with improved recall.

## 🎯 Overview

Gray-Tunneled Hashing (GTH) is a distribution-aware hashing method that treats binary code assignment as an explicit optimization problem. Unlike traditional approaches that assign codes via simple quantization, GTH optimizes the mapping to align Hamming distances in binary space with semantic distances in the original embedding space.

### Key Results

**Sprint 8:**
- **GTH outperforms baselines in 7/8 configurations** with recall improvements of **+15% to +91%**
- Best configuration achieves **8.2% recall** vs **4.3% baseline** (+90.7% improvement)
- Works particularly well with **Hyperplane LSH** (+61% to +91% improvements)

**Sprint 9 (New):**
- **Multi-radius objective**: Optimize J(φ) with multiple Hamming radii and constrained weights
- **Adaptive tunneling**: Tunneling triggered by stagnation detection or probabilistically
- **Investigation tools**: Scripts for analyzing J(φ) correlation, build time, and Hamming ball coverage

## 📚 Table of Contents

- [Theory](#theory)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Core Classes](#core-classes)
- [API Reference](#api-reference)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiments](#experiments)

## 🧮 Theory

### Problem Formulation

GTH solves a **Quadratic Assignment Problem (QAP)** where:

- **Locations**: Hypercube vertices (binary codes in `{0,1}^n`)
- **Facilities**: Embeddings from the dataset
- **Flow**: Query-neighbor co-occurrence probabilities
- **Distances**: Semantic dissimilarity (cosine distance)

**Objective Function (J(φ))**:

Single-radius:
```
J(φ) = Σ_{i,j} π_i · w_ij · E[d_H(φ(h(q)), φ(h(x))) | q∈bucket_i, x∈bucket_j]
```

Multi-radius (Sprint 9):
```
J(φ) = Σ_r w_r · J_r(φ)
where J_r(φ) considers only pairs with d_H ≤ r
```

where:
- `π_i`: Query prior for bucket `i`
- `w_ij`: Neighbor co-occurrence weight between buckets `i` and `j`
- `w_r`: Weight for radius `r` (with constraint `w_1 > w_2 > ... > 0`)
- `φ`: GTH permutation mapping bucket codes to optimized binary codes
- `h`: LSH encoder mapping embeddings to bucket codes
- `d_H`: Hamming distance

### Key Insight

The assignment of embeddings to hypercube vertices matters critically. If semantically similar embeddings are mapped to Hamming-nearby codes, a single bit flip results in limited distortion, improving search quality.

**Visualization**:

```
Embedding Space (ℝᵈ)          Hypercube Qₙ (Binary Codes)
─────────────────────          ────────────────────────────
    w₁ ●                       000 ── 001
       │                        │     │
    w₂ ● ────── semantic ────── 010 ── 011   Hamming-1 edges
       │   distance              │     │
    w₃ ●                       100 ── 101
       │                        │     │
    w₄ ●                       110 ── 111
       
Goal: Find permutation φ such that
  Hamming-1 neighbors ↔ Semantically similar embeddings
```

### Optimization

GTH uses **hill climbing with 2-swap moves** to minimize J(φ):

1. **Initialization**: Random or identity permutation
2. **Hill Climbing**: Iteratively swap bucket code assignments to reduce cost
3. **Block Tunneling** (Sprint 9): Adaptive tunneling triggered by:
   - **Stagnation detection**: When improvement < threshold over N iterations
   - **Probabilistic**: With probability `p` at each iteration
   - Reoptimizes small subsets to escape local minima

The 2-swap operator transposes the binary codes assigned to two buckets, with efficient delta computation to avoid full cost recalculation.

## 🏗️ Architecture

### High-Level Pipeline

```
1. LSH Encoding: embeddings → bucket codes
2. GTH Optimization: bucket codes → optimized binary codes
3. Index Building: create mapping from codes to dataset indices
4. Query Time: 
   - Encode query → bucket code
   - Apply GTH permutation → optimized code
   - Expand Hamming ball around optimized code
   - Retrieve candidates from matching buckets
```

### Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Query Pipeline                       │
│  query → LSH → GTH permutation → Hamming ball → results│
└─────────────────────────────────────────────────────────┘
                            ↑
                            │
┌─────────────────────────────────────────────────────────┐
│              Distribution-Aware Index                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Traffic  │→ │ J(φ)     │→ │ GTH      │            │
│  │ Stats    │  │ Objective│  │ Optimizer│            │
│  └──────────┘  └──────────┘  └──────────┘            │
└─────────────────────────────────────────────────────────┘
                            ↑
                            │
┌─────────────────────────────────────────────────────────┐
│                    LSH Encoder                          │
│  embeddings → bucket codes (Hyperplane/p-stable)        │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
gray-tunneled-hashing/
├── src/gray_tunneled_hashing/     # Main package
│   ├── algorithms/                # Core algorithms
│   │   ├── gray_tunneled_hasher.py    # Main GTH class
│   │   ├── qap_objective.py           # QAP cost computation
│   │   ├── block_moves.py              # Block tunneling
│   │   ├── block_selection.py          # Block selection strategies
│   │   └── simulated_annealing.py     # SA optimization
│   ├── distribution/              # Distribution-aware components
│   │   ├── j_phi_objective.py         # J(φ) objective (Sprint 8)
│   │   ├── cosine_objective.py         # Cosine distance objective
│   │   ├── traffic_stats.py           # Query traffic statistics
│   │   └── pipeline.py                 # End-to-end pipeline
│   ├── api/                       # Public API
│   │   ├── index_builder.py           # Index construction
│   │   └── query_pipeline.py          # Query-time pipeline
│   ├── binary/                    # Binary code utilities
│   │   ├── lsh_families.py            # LSH implementations
│   │   ├── codebooks.py               # Codebook management
│   │   └── baselines.py               # Baseline methods
│   ├── data/                      # Data handling
│   │   ├── synthetic_generators.py    # Synthetic data
│   │   └── real_datasets.py            # Real dataset loaders
│   ├── evaluation/                # Evaluation metrics
│   │   └── metrics.py                 # Recall, precision, etc.
│   └── integrations/              # External integrations
│       └── hamming_index.py           # Hamming index wrapper
├── tests/                         # Test suite
├── scripts/                       # Utility scripts
│   ├── run_sprint8_benchmark.py       # Benchmark script
│   └── analyze_sprint8_benchmark_results.py
├── experiments/                   # Experiments and results
│   └── real/
│       ├── reports/                    # Analysis reports
│       ├── results_json/               # JSON results (not versioned)
│       └── data/                        # Datasets
└── theory/                        # Theoretical documentation
    └── THEORY_AND_RESEARCH_PROGRAM.md
```

## 🔧 Core Classes

### `GrayTunneledHasher`

Main class implementing the GTH algorithm.

**Key Methods**:
- `fit_with_traffic()`: Optimize permutation using distribution-aware objective
- `get_assignment()`: Get final permutation `(K, n_bits)` mapping buckets to codes
- `encode()`: Encode embeddings to binary codes (legacy)

**Sprint 8 Changes**:
- Permutation structure changed from `(N,)` to `(K, n_bits)` where `K` is number of buckets
- New objective `J(φ)` computed over real query-neighbor pairs
- Real embeddings objective: `compute_j_phi_cost_real_embeddings()`

### `build_distribution_aware_index()`

End-to-end pipeline for building a distribution-aware index.

**Steps**:
1. Compute traffic statistics (`π`, `w`) from queries and ground truth
2. Initialize LSH encoder and create `code_to_bucket` mapping
3. Optimize GTH permutation using `J(φ)` objective
4. Build index mapping codes to dataset indices

### `query_with_hamming_ball()`

Query-time pipeline with Hamming ball expansion.

**Steps**:
1. Encode query → bucket code `c_q`
2. Apply GTH permutation → optimized code `c̃_q = φ(c_q)`
3. Expand Hamming ball: `C_q(r) = {z : d_H(z, c̃_q) ≤ r}`
4. Retrieve candidates from buckets whose permuted codes fall in ball

## 📖 API Reference

### Building an Index

```python
from gray_tunneled_hashing.api.index_builder import build_distribution_aware_index
from gray_tunneled_hashing.binary.lsh_families import create_lsh_family

# Create LSH encoder
encoder = create_lsh_family("hyperplane", n_bits=8, dim=64, random_state=42)

# Build index (Sprint 8)
index = build_distribution_aware_index(
    base_embeddings=base_embeddings,  # Shape (N, dim)
    queries=queries,                   # Shape (Q, dim)
    ground_truth_neighbors=gt_neighbors,  # Shape (Q, k)
    encoder=encoder,
    n_bits=8,
    n_codes=32,
    max_two_swap_iters=20,
)

# Build index with Sprint 9 features (multi-radius + tunneling)
index = build_distribution_aware_index(
    base_embeddings=base_embeddings,
    queries=queries,
    ground_truth_neighbors=gt_neighbors,
    encoder=encoder,
    n_bits=8,
    n_codes=32,
    max_two_swap_iters=20,
    # Sprint 9: Multi-radius objective
    hamming_radii=[1, 2, 3],  # Optimize for multiple radii
    radius_weights=None,  # Auto-generate: [1.0, 0.5, 0.25]
    # Sprint 9: Adaptive tunneling
    tunneling_on_stagnation=True,  # Enable tunneling when stagnant
    tunneling_probability=0.1,  # 10% chance per iteration
    stagnation_window=10,  # Check last 10 iterations
    stagnation_threshold=0.001,  # 0.1% improvement threshold
)

# index contains:
# - permutation: (K, n_bits) array
# - code_to_bucket: dict mapping codes to bucket indices
# - bucket_to_dataset_indices: dict mapping buckets to dataset indices
```

### Querying

```python
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball

# Query
result = query_with_hamming_ball(
    query_embedding=query,              # Shape (dim,)
    encoder=encoder,
    permutation=index["permutation"],   # Shape (K, n_bits)
    code_to_bucket=index["code_to_bucket"],
    bucket_to_dataset_indices=index["bucket_to_dataset_indices"],
    hamming_radius=1,
)

# result.candidate_indices contains candidate dataset indices
```

### Direct GTH Usage

```python
from gray_tunneled_hashing.algorithms.gray_tunneled_hasher import GrayTunneledHasher

hasher = GrayTunneledHasher(
    n_bits=8,
    max_two_swap_iters=20,
    num_tunneling_steps=0,
    mode="two_swap_only",
)

hasher.fit_with_traffic(
    queries=queries,
    base_embeddings=base_embeddings,
    ground_truth_neighbors=gt_neighbors,
    encoder=encoder,
    code_to_bucket=code_to_bucket,
    use_real_embeddings_objective=True,
    # Sprint 9: Multi-radius objective
    hamming_radii=[1, 2, 3],
    radius_weights=None,  # Auto-generate weights
    # Sprint 9: Adaptive tunneling
    tunneling_on_stagnation=True,
    tunneling_probability=0.1,
    stagnation_window=10,
    stagnation_threshold=0.001,
)

permutation = hasher.get_assignment()  # Shape (K, n_bits)
```

## 🚀 Installation

### Requirements

- Python >= 3.10
- numpy
- scipy (for some LSH families)
- tqdm (for progress bars)

### Setup

```bash
git clone https://github.com/crbazevedo/gray-tunneled-hashing.git
cd gray-tunneled-hashing
pip install -e .
```

## 🎬 Quick Start

### Running the Benchmark

**Sprint 8 benchmark:**
```bash
python scripts/run_sprint8_benchmark.py \
    --dataset synthetic \
    --n-bits 6,8 \
    --n-codes 16,32 \
    --k 10 \
    --hamming-radius 1,2 \
    --max-iters 10,20 \
    --output experiments/real/results_sprint8.json
```

**Sprint 9 benchmark (with tunneling and multi-radius):**
```bash
python scripts/run_sprint8_benchmark.py \
    --dataset synthetic \
    --n-bits 6,8 \
    --n-codes 16,32 \
    --k 10 \
    --hamming-radius 1,2 \
    --max-iters 10,20 \
    --hamming-radii 1,2,3 \
    --tunneling-on-stagnation \
    --tunneling-probability 0.1 \
    --stagnation-window 10 \
    --output experiments/real/results_sprint9.json
```

### Analyzing Results

```bash
python scripts/analyze_sprint8_benchmark_results.py \
    --input experiments/real/results_sprint8.json \
    --output experiments/real/reports/analysis.md
```

### Sprint 9 Investigation Tools

**J(φ) Correlation Analysis:**
```bash
python scripts/investigate_jphi_correlation.py \
    --benchmark-results experiments/real/results_sprint8.json \
    --output-dir experiments/real/reports \
    --generate-data
```

**Build Time Profiling:**
```bash
python scripts/profile_build_time.py \
    --n-base 1000 \
    --n-queries 100 \
    --n-bits 8 \
    --n-codes 32 \
    --output-dir experiments/real/reports
```

**Hamming Ball Coverage Analysis:**
```bash
python scripts/analyze_hamming_ball_coverage.py \
    --n-base 1000 \
    --n-queries 100 \
    --n-bits 8 \
    --n-codes 32 \
    --radii 1,2,3,4 \
    --output-dir experiments/real/reports
```

### Basic Usage Example

```python
import numpy as np
from gray_tunneled_hashing.api.index_builder import build_distribution_aware_index
from gray_tunneled_hashing.binary.lsh_families import create_lsh_family
from gray_tunneled_hashing.api.query_pipeline import query_with_hamming_ball

# Generate synthetic data
np.random.seed(42)
base_embeddings = np.random.randn(1000, 64)
queries = np.random.randn(100, 64)

# Compute ground truth (simplified)
from sklearn.metrics.pairwise import cosine_similarity
gt_neighbors = cosine_similarity(queries, base_embeddings).argsort(axis=1)[:, -10:]

# Create encoder
encoder = create_lsh_family("hyperplane", n_bits=8, dim=64, random_state=42)

# Build index
index = build_distribution_aware_index(
    base_embeddings=base_embeddings,
    queries=queries,
    ground_truth_neighbors=gt_neighbors,
    encoder=encoder,
    n_bits=8,
    n_codes=32,
    max_two_swap_iters=10,
    hamming_radius=1,
)

# Query
result = query_with_hamming_ball(
    query_embedding=queries[0],
    encoder=encoder,
    permutation=index["permutation"],
    code_to_bucket=index["code_to_bucket"],
    bucket_to_dataset_indices=index["bucket_to_dataset_indices"],
    hamming_radius=1,
)

print(f"Found {len(result.candidate_indices)} candidates")
```

## 🧪 Experiments

### Results Location

- **Reports**: `experiments/real/reports/`**
  - `SPRINT8_BENCHMARK_RESULTS_REPORT.md`: Complete analysis
  - `SPRINT8_BENCHMARK_ANALYSIS.md`: Automated analysis
  - `RECALL_RESULTS_SUMMARY.md`: Historical recall summary
- **JSON Results**: `experiments/real/results_json/` (not versioned)

### Key Findings (Sprint 8)

1. **GTH outperforms baselines** in 7/8 configurations
2. **Hyperplane LSH** works best with GTH (+61% to +91% improvements)
3. **p-stable LSH** shows smaller gains (+11% to +36%)
4. **J(φ) objective** successfully optimizes for recall on real embeddings

### Known Issues

1. **J(φ) correlation**: For `n_bits=8`, J(φ) worsens but recall improves (investigating)
2. **Build time**: ~100s per configuration (optimization opportunity)
3. **Hamming ball coverage**: Low (1-8%), may benefit from larger radius

## 📚 Documentation

- **Theory**: `theory/THEORY_AND_RESEARCH_PROGRAM.md`
- **Development Notes**: `project_management/instructions/DEVELOPMENT_NOTES.md`
- **Sprint Log**: `project_management/sprints/sprint-log.md`
- **Experiments**: `experiments/real/README.md`

## 🤝 Contributing

See `project_management/instructions/CONTRIBUTING.md` for guidelines.

## 📄 License

[Add license information]

## 🙏 Acknowledgments

[Add acknowledgments]

---

**Status**: Active development - Sprint 8 completed  
**Last Updated**: 2025-01-27
