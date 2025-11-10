# Guia de Execução: Benchmarks Distribution-Aware GTH

## Benchmarks Disponíveis

### 1. Benchmark Teórico (`benchmark_distribution_aware_theoretical.py`)

Valida garantias teóricas e compara métodos em datasets sintéticos.

**Uso**:
```bash
python scripts/benchmark_distribution_aware_theoretical.py \
    --n-bits 10 \
    --n-codes 32 \
    --k 5 \
    --traffic-scenario skewed \
    --n-runs 5 \
    --mode two_swap_only
```

**Cenários de Tráfego**:
- `uniform`: Distribuição uniforme
- `skewed`: 80% queries de 20% do espaço
- `clustered`: Queries em 3-5 clusters

### 2. Benchmark com Dataset Real (`benchmark_distribution_aware.py`)

Compara canonical vs distribution-aware em datasets reais.

**Uso**:
```bash
python scripts/benchmark_distribution_aware.py \
    --dataset quora \
    --n-bits 64 \
    --n-codes 512 \
    --k 10 \
    --mode full
```

**Requisitos**:
- Dataset deve estar em `experiments/real/data/`
- Arquivos necessários:
  - `{dataset}_base_embeddings.npy`
  - `{dataset}_queries_embeddings.npy`
  - `{dataset}_ground_truth_indices.npy`

## Preparação de Dados

### Gerar Ground Truth

```bash
python scripts/compute_float_ground_truth.py \
    --dataset quora \
    --k 10
```

### Estrutura de Diretórios

```
experiments/real/
├── data/
│   ├── {dataset}_base_embeddings.npy
│   ├── {dataset}_queries_embeddings.npy
│   └── {dataset}_ground_truth_indices.npy
└── results_distribution_aware_*.json
```

## Interpretação de Resultados

### Garantia Teórica

A garantia **J(φ*) ≤ J(φ₀)** deve sempre ser satisfeita. Se violada, indica:
- Bug no cálculo de J(φ) ou J(φ₀)
- Problema na otimização (não está minimizando J)
- Erro no mapeamento bucket → código

### Métricas

- **J(φ)**: Custo do layout otimizado (menor é melhor)
- **J(φ₀)**: Custo do layout original (baseline)
- **Improvement**: (J(φ₀) - J(φ)) / J(φ₀) * 100% (positivo = melhoria)
- **Recall@k**: Fração de vizinhos verdadeiros recuperados

### Comparação de Métodos

- **canonical**: GTH padrão (apenas distâncias semânticas)
- **distribution_aware_semantic**: Distribution-aware com distâncias semânticas
- **distribution_aware_pure**: Distribution-aware sem distâncias semânticas

Espera-se que distribution-aware tenha melhor recall@k em cenários de tráfego enviesado.

## Troubleshooting

### Erro: "FileNotFoundError: Embeddings file not found"

- Verifique que os arquivos estão em `experiments/real/data/`
- Execute `scripts/compute_float_ground_truth.py` se necessário

### Erro: "J(φ*) > J(φ₀)" (violação da garantia)

- Indica bug na implementação
- Verifique:
  1. Cálculo de J(φ₀) usa códigos originais?
  2. Permutação aprendida está correta?
  3. Mapeamento bucket → código está correto?

### Performance Lenta

- Reduza `--n-bits` (ex: 10 em vez de 16)
- Reduza `--n-codes` (ex: 32 em vez de 64)
- Use `--mode two_swap_only` em vez de `full`
- Reduza `--n-runs` para testes rápidos

## Próximos Passos

1. **Corrigir validação da garantia teórica** (bug atual)
2. **Executar em datasets reais** quando disponíveis
3. **Análise de sensibilidade** (variação de parâmetros)
4. **Documentar resultados empíricos** quando obtidos

