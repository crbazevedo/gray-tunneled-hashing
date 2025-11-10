# Sprint 8 Benchmark Results

**Date**: N/A
**Dataset**: synthetic
**N_base**: 1000
**N_queries**: 100

## Summary Comparison

| LSH Family | n_bits | Radius | Baseline | GTH | Improvement | Status |
|------------|--------|--------|----------|-----|-------------|--------|
| hyperplane | 6 | 1 | 0.0410 | 0.0730 | +78.05% | ✅ |
| hyperplane | 6 | 2 | 0.0330 | 0.0600 | +81.82% | ✅ |
| hyperplane | 8 | 1 | 0.0430 | 0.0820 | +90.70% | ✅ |
| hyperplane | 8 | 2 | 0.0390 | 0.0630 | +61.54% | ✅ |
| p | 6 | 1 | 0.0140 | 0.0190 | +35.71% | ✅ |
| p | 6 | 2 | 0.0130 | 0.0150 | +15.38% | ✅ |
| p | 8 | 1 | 0.0190 | 0.0180 | -5.26% | ❌ |
| p | 8 | 2 | 0.0190 | 0.0210 | +10.53% | ✅ |

## Best GTH Configurations

| Configuration | Recall | Build Time (s) | J(φ) Improvement |
|---------------|--------|----------------|------------------|
| hyperplane_nbits8_ncodes16_k10_radius1_iters10_tunnel0_modetwo_swap_only | 0.0820 | 80.10 | -60.90% |
| hyperplane_nbits8_ncodes32_k10_radius1_iters10_tunnel0_modetwo_swap_only | 0.0820 | 78.85 | -60.90% |
| hyperplane_nbits8_ncodes16_k10_radius1_iters20_tunnel0_modetwo_swap_only | 0.0810 | 156.75 | -43.46% |
| hyperplane_nbits8_ncodes32_k10_radius1_iters20_tunnel0_modetwo_swap_only | 0.0810 | 158.03 | -43.46% |
| hyperplane_nbits6_ncodes16_k10_radius1_iters10_tunnel0_modetwo_swap_only | 0.0730 | 86.54 | +42.19% |
| hyperplane_nbits6_ncodes16_k10_radius1_iters20_tunnel0_modetwo_swap_only | 0.0730 | 91.34 | +42.19% |
| hyperplane_nbits6_ncodes32_k10_radius1_iters10_tunnel0_modetwo_swap_only | 0.0730 | 74.20 | +42.19% |
| hyperplane_nbits6_ncodes32_k10_radius1_iters20_tunnel0_modetwo_swap_only | 0.0730 | 95.07 | +42.19% |
| hyperplane_nbits8_ncodes16_k10_radius2_iters10_tunnel0_modetwo_swap_only | 0.0630 | 74.56 | -60.90% |
| hyperplane_nbits8_ncodes32_k10_radius2_iters10_tunnel0_modetwo_swap_only | 0.0630 | 84.60 | -60.90% |