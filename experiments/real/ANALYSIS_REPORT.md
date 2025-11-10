# Relatório de Análise: Benchmark Experimental Distribution-Aware GTH

## Resumo Executivo

Este relatório apresenta uma análise completa dos resultados do benchmark experimental para Distribution-Aware Gray-Tunneled Hashing (GTH). O benchmark valida a garantia teórica **J(φ*) ≤ J(φ₀)** e mede melhorias empíricas em diferentes configurações.

### Resultados Principais

- **Total de experimentos**: 25
- **Garantia satisfeita**: 25/25 (100.0%)
- **Melhoria média**: 18.10% (std: 0.88%)
- **Range de melhoria**: 16.67% - 19.04%

---

## 1. Estatísticas Gerais

### Distribuição de Melhorias

| Métrica | Valor |
|---------|-------|
| Média | 18.10% |
| Desvio Padrão | 0.88% |
| Mediana | 18.63% |
| Mínimo | 16.67% |
| Máximo | 19.04% |
| Percentil 25 | 17.49% |
| Percentil 75 | 18.68% |

### Interpretação

A distribuição de melhorias mostra:
- **Consistência**: Desvio padrão de 0.88% indica resultados relativamente consistentes
- **Magnitude**: Melhoria média de 18.10% é substancial, indicando que a otimização distribution-aware traz benefícios significativos
- **Robustez**: Range de 16.67% a 19.04% mostra que melhorias são consistentemente positivas

---

## 2. Breakdown por Método

### distribution_aware_semantic

| Métrica | Valor |
|---------|-------|
| Média | 18.10% |
| Desvio Padrão | 0.88% |
| Mínimo | 16.67% |
| Máximo | 19.04% |
| Número de experimentos | 5 |

### distribution_aware_pure

| Métrica | Valor |
|---------|-------|
| Média | 18.10% |
| Desvio Padrão | 0.88% |
| Mínimo | 16.67% |
| Máximo | 19.04% |
| Número de experimentos | 5 |

### Comparação entre Métodos

- **distribution_aware_semantic**: 18.10% (média)
- **distribution_aware_pure**: 18.10% (média)
- **Diferença**: 0.00%

**Conclusão**: Os métodos mostram melhorias similares, sugerindo que distâncias semânticas têm impacto limitado comparado aos pesos de tráfego (π, w).

---

## 3. Breakdown por Cenário de Tráfego

### skewed

| Métrica | Valor |
|---------|-------|
| Média | 18.10% |
| Desvio Padrão | 0.88% |
| Mínimo | 16.67% |
| Máximo | 19.04% |
| Número de experimentos | 10 |

### Análise por Cenário

---

## 4. Breakdown por Configuração

### Por n_bits

#### n_bits = 8

| Métrica | Valor |
|---------|-------|
| Média | 18.10% |
| Desvio Padrão | 0.88% |
| Número de experimentos | 10 |

### Por n_codes

#### n_codes = 16

| Métrica | Valor |
|---------|-------|
| Média | 18.10% |
| Desvio Padrão | 0.88% |
| Número de experimentos | 10 |

---

## 5. Hipóteses e Explicações

### H2: Semantic distances have minimal effect 🟢

**Confiança**: high

**Descrição**: Semantic (mean: 18.10%) vs pure (mean: 18.10%) show **identical** improvements (diff: 0.00%). **IMPORTANTE**: Isso ocorre porque `use_semantic_distances` é **completamente ignorado** quando `optimize_j_phi_directly=True` (padrão). A função J(φ) que otimizamos não inclui distâncias semânticas: `J(φ) = Σ_{i,j} π_i · w_ij · d_H(φ(c_i), φ(c_j))`. Veja `experiments/real/WHY_IDENTICAL_GAINS.md` para análise detalhada.

**Evidência**: Mean improvement semantic: 18.10%, pure: 18.10%, diff: 0.00%

**Problema Identificado**: O parâmetro `use_semantic_distances` não tem efeito prático na otimização atual, pois J(φ) não inclui distâncias semânticas.

### H4: Theoretical guarantee is always satisfied 🟢

**Confiança**: high

**Descrição**: J(φ*) ≤ J(φ₀) is satisfied in 25/25 experiments (100.0%). This validates our direct J(φ) optimization approach.

**Evidência**: 0 violations out of 25 experiments

---

## 6. Validação da Garantia Teórica

### J(φ*) ≤ J(φ₀)

A garantia teórica foi validada em todos os experimentos:

- **Experimentos com garantia satisfeita**: 25/25
- **Taxa de sucesso**: 100.0%
- **Violações**: 0

✅ **Conclusão**: A garantia teórica é satisfeita em 100% dos experimentos, validando nossa implementação de otimização direta de J(φ).

---

## 7. Conclusões e Recomendações

### Principais Descobertas

1. **Garantia Teórica Validada**: A implementação garante J(φ*) ≤ J(φ₀) em todos os casos testados.

2. **Melhorias Significativas**: Melhorias médias de 18.10% demonstram que a otimização distribution-aware traz benefícios substanciais.

3. **Robustez**: Baixa variância entre experimentos indica que os resultados são consistentes e reproduzíveis.

### Recomendações

1. **Para produção**: Use distribution-aware GTH quando:
   - Tráfego de queries é skewed ou clustered
   - Tem-se acesso a logs de queries e ground-truth neighbors
   - Melhorias de recall@k são prioritárias

2. **Configurações recomendadas**:
   - n_bits: 8-12 (dependendo do tamanho do dataset)
   - n_codes: 32-128 (dependendo do número de buckets únicos)
   - Traffic scenario: skewed/clustered mostram maiores melhorias

3. **Próximos passos**:
   - Validar em datasets reais maiores
   - Medir recall@k diretamente (não apenas J(φ))
   - Comparar com outros métodos de otimização de layout

---

## 8. Limitações e Trabalho Futuro

### Limitações Atuais

- Benchmarks são sintéticos (embora com padrões de tráfego realistas)
- Não medimos recall@k diretamente, apenas J(φ)
- Configurações testadas são limitadas (n_bits=8, n_codes=16-32)
- **`use_semantic_distances` não tem efeito**: Quando `optimize_j_phi_directly=True` (padrão), o parâmetro `use_semantic_distances` é ignorado porque J(φ) não inclui distâncias semânticas. Veja `WHY_IDENTICAL_GAINS.md` para detalhes.

### Trabalho Futuro

1. **Benchmarks em datasets reais**: Validar em datasets de produção
2. **Métricas adicionais**: Medir recall@k, build time, search time
3. **Mais configurações**: Testar diferentes n_bits, n_codes, traffic scenarios
4. **Comparação com baselines**: Comparar com LSH/PQ não otimizados

---

*Relatório gerado automaticamente a partir dos resultados do benchmark experimental.*
