# Gray-Tunneled Hashing: Theory and Research Program

## Overview

Gray-Tunneled Hashing is a novel approach to binary vector encoding that treats the assignment of embeddings to binary codes as an explicit optimization problem. Rather than treating binary code assignment as an afterthought of quantization, this framework models it as a Quadratic Assignment Problem (QAP) with the goal of aligning hypercube geometry (Hamming distance) with semantic geometry (embedding similarity).

## Core Problem

### Binary Vector Search Context

- **Problem**: In large-scale ANN search and vector databases, continuous embeddings are often stored as binary codes
- **Current Practice**: Binary codes are typically assigned via:
  - Simple sign-thresholding
  - Random projections
  - Product quantization where indices happen to be binary
  - Limited control over how Hamming distance relates to semantic similarity

### Key Insight

The assignment of embeddings to hypercube vertices (binary codes) matters critically. If neighboring codevectors are mapped to Hamming-nearby labels, a single bit flip or small channel error results in limited distortion. This was known in classical signal coding but is underutilized in modern large-scale embedding systems.

## Theoretical Framework

### 1. Problem Formulation

**Code Space**: The hypercube $Q_n$ with $N = 2^n$ vertices, where each vertex is a binary code in $\{0,1\}^n$.

**Assignment Problem**: Find a permutation $\pi \in S_N$ that maps embeddings $w_1, \dots, w_N$ to hypercube vertices such that:
- Hamming distance between codes correlates with semantic distance between embeddings
- Hamming-1 neighbors correspond to semantically similar embeddings

**QAP Objective**: 
\[
f(\pi) = \sum_{(u,v)\in E} d_{\pi(u)\,\pi(v)}
\]

where:
- $E$ is the set of hypercube edges (pairs of vertices with Hamming distance 1)
- $d_{ij} = \|w_i - w_j\|^2$ is the semantic distance matrix
- The goal is to minimize $f(\pi)$

This is a Quadratic Assignment Problem (QAP) with:
- **Locations**: Hypercube vertices
- **Facilities**: Embedding indices
- **Flow**: Unit weight on hypercube edges
- **Distances**: Semantic dissimilarity matrix

### 2. Elementary Landscape Theory

**Key Result**: Under 2-swap neighborhood (transpositions of two vertices), the QAP objective defines an **elementary landscape**.

**Theorem**: For the hypercube QAP under 2-swap moves:
\[
\mathbb{E}[f(\pi') \mid \pi] = \left(1 - \frac{4}{N}\right) f(\pi) + \frac{4}{N}\,\bar f
\]

where:
- $\pi'$ is a uniformly random 2-swap neighbor of $\pi$
- $\bar f$ is the global mean of $f$ over all permutations
- $\lambda = 4/N$ is the relaxation rate

**Implications**:
- Random walks under 2-swap quickly mix around the global average
- The eigenvalue $\lambda = 4/N$ becomes small as $N$ grows
- Single 2-swap steps have limited pull toward the average
- Large-scale rearrangements require many steps or more powerful moves

**Landscape Structure**:
- $f - \bar f$ is an eigenfunction of the neighbor-averaging operator with eigenvalue $1 - 4/N$
- This provides global structure but doesn't preclude many local minima
- Motivates the need for richer operators (block moves) to escape poor local minima

### 3. Planted Model and Statistical Assumptions

**Ideal Pseudo-Gray Configuration**: A map $\phi: V \to \mathbb{R}^d$ such that:
- If $\|u-v\|_H = 1$ (Hamming-1 neighbors), then $\|\phi(u) - \phi(v)\|$ is small
- If $\|u-v\|_H \ge 2$, distances are typically larger with positive margin

**Planted Structure**: There exists a planted permutation $\pi^*$ such that:
\[
w_{\pi^*(u)} = \phi(u) + \xi_u, \qquad u \in V
\]

where $\xi_u$ are independent subgaussian noise vectors with variance proxy $\sigma^2$.

**Margin Conditions**:
- Constants $0 < \delta_1 < \delta_2$ exist such that:
  - Hamming-1 neighbors: $\|\phi(u) - \phi(v)\|^2 \le \delta_1$
  - Hamming-2+ neighbors: $\|\phi(u) - \phi(v)\|^2 \ge \delta_2$
- Noise is small relative to margins: $\sigma^2 \ll \delta_2 - \delta_1$

**Expected Properties**:
- $\pi^*$ should be (almost) globally optimal
- Assignments disagreeing with $\pi^*$ on nontrivial fractions of vertices incur significant cost penalties

### 4. Statistical Tunneling via Block Moves

**Problem**: 2-swap moves alone may get trapped in poor local minima. We need operators that can "tunnel" across barriers.

**Block Moves**: Reoptimize assignments on small subsets of vertices:
1. Choose a block $B \subset V$ with $|B| = k$
2. Let $I_B$ be embeddings currently assigned to vertices in $B$
3. Solve the restricted QAP for reassigning $I_B$ to $B$
4. Apply the reassignment if it improves the global objective

**Block Types**:
- **Cluster-based**: Partition embeddings into clusters, assign cluster vertices to blocks
- **Geometric subcubes**: Choose subsets forming small subcubes of dimension $m$ ($|B| = 2^m$)
- **Hybrid**: Clusters with geometrically close vertices on hypercube

**Statistical Tunneling Conjecture**:

Under the planted pseudo-Gray model with suitable conditions, there exist constants $\varepsilon > 0$, $c > 0$, and block size $k(N) = O(\log^c N)$ such that:

1. **Near-optimality of block-local minima**: Any assignment $\pi$ that is a local minimum under block moves satisfies:
   \[
   f(\pi) - f(\pi^*) \le \varepsilon N
   \]

2. **Polynomial-time convergence**: A randomized local-search algorithm alternating 2-swap hill-climbing and randomized block reoptimizations reaches near-optimal assignments in polynomial time with high probability.

**Intuition**: Deep bad local minima are rare in typical planted instances once we allow block moves. Any persistent local minimum is near-globally optimal.

## Algorithmic Framework: Gray-Tunneled Hashing

### Practical Algorithm Sketch

1. **Initialization**
   - Obtain codebook vectors $c_1, \dots, c_N$ (e.g., via k-means, PQ)
   - Construct initial assignment $\pi_0$:
     - Sort codebooks by principal component, assign in Gray-code order
     - Or use heuristic mapping respecting locality

2. **2-Swap Local Optimization**
   - Run local search applying improving 2-swaps
   - Continue until no improvement or budget reached

3. **Block Tunneling**
   - Identify candidate blocks (clusters or geometric subcubes)
   - For each block, solve restricted QAP and apply best improving reassignment
   - Iterate between 2-swap local search and block tuning

4. **Embedding Assignment and Indexing**
   - Assign codebook indices to binary codes according to optimized $\pi$
   - For each embedding $x$, find nearest codebook vector $c_i$, get binary code via $\pi^{-1}(i)$
   - Store binary codes in Hamming-based ANN index

5. **ANN Queries**
   - For query embedding $q$, compute its code
   - Perform Hamming-distance search
   - Optionally re-rank top candidates using float embeddings

### Expected Benefits

Compared to naive binary coding:
- **Better locality**: Hamming-1 neighbors correspond more consistently to nearest codebooks
- **Higher recall**: Code geometry better aligned with embedding geometry → Hamming balls contain more true neighbors
- **Fewer bits for same quality**: Pseudo-Gray structure may allow shorter codes while maintaining target recall

All achieved without changing underlying Hamming-based infrastructure of vector databases.

## Research and Experimental Program

### Phase I: Synthetic Landscapes

**Goals**:
- Validate theoretical properties on synthetic instances
- Understand landscape structure

**Tasks**:
- Implement synthetic planted QAP instances with known $\pi^*$
- Measure distribution of local minima under 2-swap
- Evaluate effect of block moves on reaching near-$\pi^*$ assignments
- Explore how noise variance and margins affect depth of bad local minima

### Phase II: Real Embeddings and Code Assignment

**Goals**:
- Apply to real-world datasets
- Benchmark against baselines

**Tasks**:
- Use real ANN datasets with precomputed embeddings
- Learn codebooks (k-means, PQ) then optimize code assignment
- Benchmark against:
  - Sign hashing
  - Off-the-shelf binary quantization
  - Random code assignments
- Report metrics: recall@k vs bits vs latency

### Phase III: End-to-End Learning

**Goals**:
- Integrate into learning frameworks
- Evaluate downstream performance

**Tasks**:
- Incorporate pseudo-Gray regularization into autoencoders or VQ-VAE frameworks
- Optimize code assignment during training
- Evaluate downstream performance in retrieval and robustness

## Key Theoretical Questions

1. **Formalize planted model**: Make precise statements about cost gaps and local-minimum structure
2. **Prove tunneling theorem**: Rigorously establish polynomial-time convergence for block operators
3. **Quantify benefits**: Analytically characterize expected improvements in recall/quality
4. **Parameter selection**: Understand optimal block sizes, noise thresholds, and margin conditions

## Implementation Considerations

- **Scalability**: Design efficient algorithms for large $N$ (millions of embeddings)
- **Integration**: Plug into existing binary ANN infrastructure (FAISS, etc.)
- **Hyperparameters**: Tune block sizes, iteration budgets, convergence criteria
- **Parallelization**: Exploit block independence for parallel processing

## Conclusion

Gray-Tunneled Hashing transforms code assignment from a heuristic detail into a controlled combinatorial optimization problem. The framework combines:
- Rigorous QAP formulation
- Elementary landscape structure (explicit eigenvalue $\lambda = 4/N$)
- Planted pseudo-Gray model with statistical guarantees
- Block-move tunneling for escaping local minima

The research program aims to validate the theory and deploy practical algorithms in real vector database settings, with direct implications for large-scale retrieval systems.

