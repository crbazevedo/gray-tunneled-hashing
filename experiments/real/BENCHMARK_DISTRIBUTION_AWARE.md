# Benchmark Teórico: Distribution-Aware GTH

## Objetivo

Este benchmark valida teoricamente a implementação de Distribution-Aware GTH, testando:

1. **Garantia Teórica**: J(φ*) ≤ J(φ₀)
2. **Melhorias de Recall@k** sob tráfego enviesado
3. **Comparação**: Canonical vs Distribution-Aware GTH
4. **Efeito de distâncias semânticas** na matriz ponderada

## Design Teórico

### Cenários de Tráfego Controlados

O benchmark gera datasets sintéticos com padrões de tráfego controlados:

- **Uniform**: Distribuição uniforme de queries
- **Skewed**: 80% das queries vêm de 20% do espaço (distribuição Pareto-like)
- **Clustered**: Queries concentradas em 3-5 clusters específicos

### Validação Estatística

- Múltiplas execuções (n_runs) para robustez
- Validação da garantia teórica J(φ*) ≤ J(φ₀)
- Estatísticas de melhoria (mean, std, min, max)

### Métricas

1. **J(φ)**: Custo distribution-aware do layout otimizado
2. **J(φ₀)**: Custo distribution-aware do layout original (baseline)
3. **Improvement**: (J(φ₀) - J(φ)) / J(φ₀) * 100%
4. **Recall@k**: Fração de vizinhos verdadeiros recuperados (quando implementado)

## Como Executar

```bash
python scripts/benchmark_distribution_aware_theoretical.py \
    --n-bits 10 \
    --n-codes 32 \
    --k 5 \
    --traffic-scenario skewed \
    --n-runs 5 \
    --mode two_swap_only \
    --output experiments/real/results_distribution_aware_theoretical.json
```

### Parâmetros

- `--n-bits`: Número de bits (default: 16)
- `--n-codes`: Número de vetores do codebook (default: 64)
- `--k`: Número de vizinhos (default: 10)
- `--traffic-scenario`: Cenário de tráfego (uniform, skewed, clustered)
- `--n-runs`: Número de execuções para robustez (default: 5)
- `--mode`: Modo de otimização (trivial, two_swap_only, full)
- `--random-state`: Seed aleatória (default: 42)

## Resultados Esperados

### Garantia Teórica

A garantia J(φ*) ≤ J(φ₀) **deve sempre ser satisfeita**, pois:

- O conjunto de permutações é finito
- φ₀ (layout original) é uma permutação válida
- φ* = argmin_φ J(φ) minimiza J sobre todas as permutações
- Portanto: J(φ*) ≤ J(φ₀) por definição

### Melhorias Esperadas

- **Tráfego Uniform**: Melhorias pequenas ou nulas (distribution-aware ≈ canonical)
- **Tráfego Skewed**: Melhorias significativas (distribution-aware >> canonical)
- **Tráfego Clustered**: Melhorias moderadas a grandes

### Efeito de Distâncias Semânticas

- **Com semântica**: D_weighted[i,j] = π_i · w_ij · d_semantic(i,j)
- **Sem semântica**: D_weighted[i,j] = π_i · w_ij

Espera-se que incluir distâncias semânticas melhore a qualidade do layout.

## Status Atual

### Implementação

✅ **Completo**:
- Geração de datasets sintéticos com cenários controlados
- Cálculo de J(φ) e J(φ₀)
- Validação da garantia teórica
- Estatísticas de melhoria
- Comparação entre métodos

⚠️ **Em Investigação**:
- Validação empírica da garantia J(φ*) ≤ J(φ₀) está mostrando violações
- Possíveis causas:
  1. Bug no mapeamento de buckets para códigos permutados
  2. Otimização não está realmente minimizando J(φ)
  3. Cálculo de J(φ₀) pode estar incorreto

### Próximos Passos

1. **Debug do cálculo de J(φ)**:
   - Verificar mapeamento bucket → código permutado
   - Validar que a permutação aprendida está correta
   - Confirmar que J(φ₀) usa os códigos originais

2. **Validação com dataset real**:
   - Quando dados estiverem disponíveis
   - Comparar recall@k entre canonical e distribution-aware
   - Validar melhorias em cenários de tráfego enviesado

3. **Análise de sensibilidade**:
   - Variação de n_bits, n_codes
   - Efeito de diferentes cenários de tráfego
   - Impacto de incluir/excluir distâncias semânticas

## Estrutura do Output

O arquivo JSON de resultados contém:

```json
{
  "args": {
    "n_bits": 10,
    "n_codes": 32,
    "k": 5,
    "traffic_scenario": "skewed",
    "n_runs": 5,
    "mode": "two_swap_only"
  },
  "results": [
    {
      "method": "canonical",
      "recall_at_k": 0.75,
      "build_time": 2.3,
      "search_time": 0.01,
      "qap_cost": 1234.5,
      "n_bits": 10,
      "n_codes": 32,
      "mode": "two_swap_only"
    },
    {
      "method": "distribution_aware_semantic",
      "recall_at_k": 0.0,
      "build_time": 3.1,
      "search_time": 0.0,
      "j_phi": 1.644,
      "j_phi_0": 0.462,
      "j_phi_improvement": -255.84,
      "n_bits": 10,
      "n_codes": 32,
      "mode": "two_swap_only",
      "use_semantic_distances": true
    }
  ],
  "validation": {
    "guarantee_holds": false,
    "violations": [...],
    "statistics": {
      "mean_improvement": -302.51,
      "std_improvement": 46.67,
      "min_improvement": -349.18,
      "max_improvement": -255.84,
      "n_experiments": 4
    }
  }
}
```

## Referências Teóricas

- **Objetivo Distribution-Aware**: J(φ) = Σ_{i,j} π_i · w_ij · d_H(φ(c_i), φ(c_j))
- **Garantia**: J(φ*) ≤ J(φ₀) onde φ* = argmin_φ J(φ)
- **Conexão com Recall@k**: Reduzir J(φ) desloca massa de w_ij para distâncias menores, melhorando recall@k

Veja `theory/THEORY_AND_RESEARCH_PROGRAM.md` seção "Distribution-Aware Hypercube QAP" para detalhes completos.

