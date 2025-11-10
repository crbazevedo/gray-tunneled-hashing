# Resumo Consolidado de Resultados de Recall e Comparações com Baselines

**Data**: 2025-01-27  
**Status**: Resultados Atualizados até Sprint 8

## 📊 Resumo Executivo

### Situação Atual

**GTH ainda está pior que baselines em todos os experimentos realizados:**

| Sprint | Baseline Recall | GTH Recall | Diferença | Status |
|--------|----------------|------------|-----------|--------|
| Sprint 5 | 0.20 (20%) | 0.067 (6.7%) | **-66.5%** | ⚠️ GTH pior |
| Sprint 6 | 0.122 (12.2%) | 0.068 (6.8%) | **-44.3%** | ⚠️ GTH pior |
| Sprint 7 | 0.13 (13%) | 0.02-0.028 (2-2.8%) | **-78.5% a -84.6%** | ⚠️ GTH muito pior |
| Sprint 8 | - | - | - | ⏳ **Aguardando execução de benchmark** |

## 📁 Localização dos Resultados

### Arquivos Principais

1. **`experiments/real/RECALL_INVESTIGATION_REPORT.md`**
   - Investigação detalhada de baixo recall (H1-H5)
   - Identificação de 6 problemas críticos
   - **Última atualização**: Após Fix 1 e Fix 4

2. **`experiments/real/results_sprint6_experiment1_summary.md`**
   - Resultados com diferentes raios Hamming (0, 1, 2)
   - Comparação baseline vs GTH
   - **Última atualização**: Sprint 6

3. **`experiments/real/SPRINT7_DIAGNOSTIC_REPORT.md`**
   - Diagnóstico completo com múltiplos métodos de otimização
   - Análise de cobertura de Hamming ball
   - **Última atualização**: Sprint 7

4. **`experiments/real/STATUS_ANALYSIS.md`**
   - Análise crítica do estado atual
   - Problemas fundamentais identificados
   - **Última atualização**: Sprint 7

5. **`experiments/real/DIAGNOSTIC_RESULTS_SUMMARY.md`**
   - Resumo dos diagnósticos executados
   - Cobertura de Hamming ball: 10.6%
   - **Última atualização**: Sprint 7

### Arquivos JSON com Resultados Numéricos

- `experiments/real/results_sprint5_experiment1.json`
- `experiments/real/results_sprint6_experiment1_radius0.json`
- `experiments/real/results_sprint6_experiment1_radius1.json`
- `experiments/real/results_sprint6_experiment1_radius2.json`
- `experiments/real/recall_optimization_comparison.json`
- `experiments/real/optimization_methods_comparison.json`
- `experiments/real/test_recall_after_all_fixes.json`

## 📈 Resultados Detalhados por Sprint

### Sprint 5: Primeira Comparação (n_bits=6, n_codes=16, radius=1)

**Configuração**:
- n_bits: 6
- n_codes: 16
- n_samples: 50-100
- k: 3-5
- hamming_radius: 1

**Resultados**:

| Método | Recall | vs Baseline |
|--------|--------|-------------|
| baseline_hyperplane | **0.2000** | - |
| baseline_p_stable | 0.1333 | - |
| baseline_random_proj | **0.2000** | - |
| hyperplane (GTH) | 0.0667 | **-66.5%** |
| p_stable (GTH) | 0.0000 | **-100%** |
| random_proj (GTH) | 0.0000 | **-100%** |

**Observações**:
- GTH pior que baseline em todos os casos
- p_stable e random_proj com GTH tiveram recall zero
- Build time: 10-30s vs <1s (baseline)

### Sprint 6: Validação com Múltiplos Raios (n_bits=6, n_codes=16)

**Configuração**:
- n_bits: 6
- n_codes: 16
- n_samples: 100
- n_queries: 20
- k: 5
- n_runs: 5
- hamming_radius: 0, 1, 2

**Resultados por Radius**:

#### Radius 0 (Exact Match)

| Método | Recall (mean ± std) |
|--------|---------------------|
| baseline_hyperplane | **0.1220 ± 0.0172** |
| baseline_p_stable | 0.0460 ± 0.0136 |
| baseline_random_proj | **0.1220 ± 0.0172** |
| hyperplane (GTH) | 0.0220 ± 0.0194 |
| p_stable (GTH) | 0.0120 ± 0.0098 |
| random_proj (GTH) | 0.0180 ± 0.0194 |

#### Radius 1

| Método | Recall (mean ± std) |
|--------|---------------------|
| baseline_hyperplane | **0.1220 ± 0.0172** |
| baseline_p_stable | 0.0460 ± 0.0136 |
| baseline_random_proj | **0.1220 ± 0.0172** |
| hyperplane (GTH) | **0.0680 ± 0.0293** |
| p_stable (GTH) | 0.0520 ± 0.0194 |
| random_proj (GTH) | 0.0520 ± 0.0232 |

**Melhoria com Hamming Ball**:
- hyperplane: 0.022 → 0.068 (3.1x)
- p_stable: 0.012 → 0.052 (4.3x)
- random_proj: 0.018 → 0.052 (2.9x)

**Ainda pior que baseline**: -44.3% (0.068 vs 0.122)

#### Radius 2

| Método | Recall (mean ± std) |
|--------|---------------------|
| baseline_hyperplane | **0.1220 ± 0.0172** |
| baseline_p_stable | 0.0460 ± 0.0136 |
| baseline_random_proj | **0.1220 ± 0.0172** |
| hyperplane (GTH) | 0.0520 ± 0.0075 |
| p_stable (GTH) | 0.0560 ± 0.0185 |
| random_proj (GTH) | 0.0500 ± 0.0261 |

**Observação**: Radius 2 não melhorou (piorou ligeiramente)

### Sprint 7: Diagnóstico Completo (Após Fixes)

**Configuração**:
- n_bits: 6
- n_codes: 16-32
- n_samples: 100
- n_queries: 50
- k: 10
- hamming_radius: 1

**Resultados Após Fixes**:

| Método | Recall | vs Baseline |
|--------|--------|-------------|
| Baseline Hyperplane | **0.13 (13%)** | - |
| GTH Hyperplane (Fix 1+4) | 0.08 (8%) | **-38.5%** |
| GTH Hyperplane (Hill Climb J(φ)) | 0.026 (2.6%) | **-80%** |
| GTH Hyperplane (SA Cosine) | 0.028 (2.8%) | **-78.5%** |
| GTH Hyperplane (Memetic) | 0.016 (1.6%) | **-87.7%** |

**Problemas Identificados**:

1. **Cobertura de Hamming Ball Muito Baixa**: Apenas 10.6% dos neighbors estão no ball (radius=1)
2. **Correlação Cosine-Hamming Fraca**: 0.17 (muito baixa)
3. **J(φ) não melhora recall**: J(φ) melhora 12.2%, mas recall permanece 0.02
4. **Otimização direta de recall piora**: Recall surrogate melhorou, mas recall real piorou

### Sprint 8: Mudanças Estruturais (Sem Resultados de Benchmark Ainda)

**Mudanças Implementadas**:
- ✅ Nova estrutura de permutação: `(K, n_bits)` em vez de `(N,)`
- ✅ Objetivo J(φ) sobre embeddings reais
- ✅ Query pipeline corrigido (permutação antes de Hamming ball)
- ✅ 69 testes implementados e passando

**Status**: ⏳ **Aguardando execução de benchmark completo com dados reais**

## 🔍 Problemas Fundamentais Identificados

### 1. Cobertura de Hamming Ball Insuficiente

**Evidência**:
- Radius=1: Apenas 10.6% dos neighbors no ball
- Radius=2: 29.8% dos neighbors no ball
- Radius=3: 63.2% dos neighbors no ball

**Impacto**: Mesmo com radius=3, 36.8% dos neighbors não são cobertos

### 2. Correlação Cosine-Hamming Muito Fraca

**Evidência**:
- Correlação Pearson: 0.17 (muito baixa)
- GTH não melhora correlação (permanece 0.17)

**Impacto**: Otimizar distâncias Hamming não melhora recall (que depende de cosine)

### 3. J(φ) Não É Proxy Adequado para Recall

**Evidência**:
- J(φ) melhora 12.2% (2.618 → 2.298)
- Recall não muda (0.02 → 0.02)
- Correlação J(φ)-recall: 0.42 (p=0.30, não significativa)

**Impacto**: Otimizar J(φ) não melhora recall

### 4. Otimização Direta de Recall Piora

**Evidência**:
- J(φ) optimization recall: 0.032
- Recall optimization recall: 0.018
- **Piorou**: -0.014

**Impacto**: Problema não é apenas a função objetivo, mas estrutura do espaço de busca

## 📊 Comparação Consolidada

### Melhor Resultado GTH vs Baseline

| Sprint | Melhor GTH | Baseline | Diferença | Método GTH |
|--------|------------|----------|-----------|------------|
| Sprint 5 | 0.0667 | 0.2000 | **-66.5%** | hyperplane |
| Sprint 6 | 0.0680 | 0.1220 | **-44.3%** | hyperplane (radius=1) |
| Sprint 7 | 0.0800 | 0.1300 | **-38.5%** | hyperplane (após Fix 1+4) |
| Sprint 7 | 0.0280 | 0.1300 | **-78.5%** | SA Cosine |

**Tendência**: Melhorou de -66.5% para -38.5% após fixes, mas ainda muito abaixo

### Métodos de Otimização Comparados (Sprint 7)

| Método | J(φ) Cost | Recall | Tempo (s) |
|--------|-----------|--------|-----------|
| Hill Climb (J(φ)) | 2.272 | 0.026 | 229 |
| Simulated Annealing (J(φ)) | 2.162 | 0.014 | 713 |
| Memetic Algorithm (J(φ)) | 2.128 | 0.016 | 4656 |
| Hill Climb (Cosine) | 2.224 | 0.018 | 356 |
| **Simulated Annealing (Cosine)** | **2.156** | **0.028** | **1133** |

**Melhor**: SA com Cosine Objective (0.028), mas ainda 78.5% pior que baseline

## 🎯 Conclusões Principais

1. **GTH está consistentemente pior que baselines** em todos os experimentos
2. **Fixes estruturais melhoraram** de -66.5% para -38.5%, mas ainda insuficiente
3. **Problema fundamental**: J(φ) não é proxy adequado para recall
4. **Cobertura de Hamming ball muito baixa**: Apenas 10.6% com radius=1
5. **Correlação cosine-Hamming fraca**: 0.17 (muito baixa)
6. **Sprint 8 implementou mudanças estruturais**, mas **não há resultados de benchmark ainda**

## 📝 Próximos Passos

1. ⏳ **Executar benchmark completo da Sprint 8** com dados reais
2. ⏳ **Validar se nova estrutura (K, n_bits) melhora recall**
3. ⏳ **Testar objetivo J(φ) sobre embeddings reais** vs. objetivo teórico
4. ⏳ **Comparar com baselines** usando novos testes comparativos
5. ⏳ **Analisar se recall melhorou** após mudanças da Sprint 8

## 📂 Arquivos de Referência

- **Resultados numéricos**: `experiments/real/*.json`
- **Relatórios de análise**: `experiments/real/*_REPORT.md`, `*_SUMMARY.md`
- **Scripts de benchmark**: `scripts/benchmark_*.py`, `scripts/run_*_experiment*.py`
- **Testes comparativos**: `tests/test_sprint8_recall_comparative.py`

