# Sprint 8 - Plano de Execução do Benchmark Completo com Dados Reais

**Data**: 2025-01-27  
**Status**: Planejamento Completo

## Objetivo

Executar benchmark completo comparando GTH (Sprint 8) com baselines usando dados reais, validando se as mudanças estruturais da Sprint 8 melhoram o recall.

## Mudanças da Sprint 8 a Validar

1. ✅ **Nova estrutura de permutação**: `(K, n_bits)` em vez de `(N,)`
2. ✅ **Objetivo J(φ) sobre embeddings reais**: Usa pares query-neighbor reais
3. ✅ **Query pipeline corrigido**: Permutação aplicada antes de Hamming ball
4. ✅ **Integração completa**: `build_distribution_aware_index` usa novo objetivo por padrão

## Configuração do Benchmark

### Datasets

**Dataset Principal**: Usar dataset real disponível em `experiments/real/data/`

**Opções**:
1. Dataset sintético (se não houver real): `generate_synthetic_dataset_for_benchmark.py`
2. Dataset real (se disponível): Verificar `experiments/real/data/*.npy`

**Tamanho Recomendado**:
- Base embeddings: N = 1000-5000 (para benchmark rápido)
- Queries: Q = 100-500
- k (recall@k): 10

### Configurações a Testar

#### 1. Configuração Base (Validação Rápida)

```yaml
n_bits: [6, 8]
n_codes: [16, 32]
k: 10
hamming_radius: [1, 2]
max_two_swap_iters: [10, 20]
num_tunneling_steps: 0
mode: "two_swap_only"
random_state: 42
n_runs: 3  # Para média e desvio padrão
```

#### 2. Configuração Completa (Validação Extensiva)

```yaml
n_bits: [6, 8, 10]
n_codes: [16, 32, 64]
k: [5, 10, 20]
hamming_radius: [1, 2, 3]
max_two_swap_iters: [10, 20, 50]
num_tunneling_steps: [0, 5, 10]
mode: ["two_swap_only", "full"]
random_state: [42, 43, 44]  # Para múltiplas seeds
n_runs: 5
```

### Métodos a Comparar

#### Baselines (Sem GTH)

1. **Baseline Hyperplane LSH**
   - LSH Hyperplane sem otimização
   - Hamming ball search com radius=1,2,3

2. **Baseline p-stable LSH**
   - LSH p-stable sem otimização
   - Hamming ball search com radius=1,2,3

3. **Baseline Random Projection**
   - Random projection sem otimização
   - Hamming ball search com radius=1,2,3

#### GTH (Sprint 8)

1. **GTH Hyperplane (Sprint 8)**
   - LSH Hyperplane + GTH com nova estrutura
   - Objetivo J(φ) sobre embeddings reais
   - Hamming ball search com radius=1,2,3

2. **GTH p-stable (Sprint 8)**
   - LSH p-stable + GTH com nova estrutura
   - Objetivo J(φ) sobre embeddings reais
   - Hamming ball search com radius=1,2,3

### Métricas a Coletar

Para cada método e configuração:

1. **Recall@k**: Fração de neighbors verdadeiros recuperados
2. **Build Time**: Tempo de construção do índice (s)
3. **Search Time**: Tempo total de busca (s)
4. **Avg Search Time**: Tempo médio por query (ms)
5. **J(φ) Cost**: Custo do objetivo J(φ) (se aplicável)
6. **J(φ) Improvement**: Melhoria em relação ao inicial (%)
7. **Hamming Ball Coverage**: % de neighbors no ball
8. **Candidates per Query**: Número médio de candidatos por query
9. **Memory Usage**: Uso de memória (MB)

## Estrutura do Script

### Script Principal: `scripts/run_sprint8_benchmark.py`

**Funcionalidades**:
1. Carregar dados reais (ou gerar sintéticos)
2. Executar baselines (Hyperplane, p-stable, Random Proj)
3. Executar GTH (Sprint 8) com diferentes configurações
4. Coletar métricas detalhadas
5. Gerar relatório comparativo
6. Salvar resultados em JSON

**Parâmetros**:
```bash
--dataset DATASET_NAME          # Nome do dataset (ou "synthetic")
--n-bits N_BITS                 # Lista de n_bits: 6,8,10
--n-codes N_CODES               # Lista de n_codes: 16,32,64
--k K                           # Lista de k: 5,10,20
--hamming-radius RADIUS         # Lista de radius: 1,2,3
--max-iters MAX_ITERS           # Lista de max_two_swap_iters: 10,20,50
--tunneling-steps STEPS         # Lista de num_tunneling_steps: 0,5,10
--mode MODE                     # two_swap_only ou full
--n-runs N_RUNS                 # Número de runs para média (default: 3)
--random-state SEED             # Seed inicial (default: 42)
--output OUTPUT_FILE            # Arquivo de saída JSON
--quick                         # Modo rápido (configuração base apenas)
```

## Plano de Execução

### Fase 1: Preparação (30 min)

1. ✅ Verificar disponibilidade de dados reais
2. ✅ Gerar ground truth se necessário
3. ✅ Validar que scripts funcionam com Sprint 8
4. ✅ Criar script de benchmark

### Fase 2: Validação Rápida (1-2 horas)

**Objetivo**: Validar que tudo funciona e obter resultados iniciais

**Configuração**:
- n_bits: [6, 8]
- n_codes: [16, 32]
- k: 10
- hamming_radius: [1, 2]
- max_two_swap_iters: [10, 20]
- num_tunneling_steps: 0
- mode: "two_swap_only"
- n_runs: 3

**Comando**:
```bash
python scripts/run_sprint8_benchmark.py \
    --dataset synthetic \
    --n-bits 6,8 \
    --n-codes 16,32 \
    --k 10 \
    --hamming-radius 1,2 \
    --max-iters 10,20 \
    --tunneling-steps 0 \
    --mode two_swap_only \
    --n-runs 3 \
    --quick \
    --output experiments/real/results_sprint8_quick.json
```

**Resultados Esperados**:
- Comparação GTH vs baselines
- Validação de que recall melhorou (ou não)
- Identificação de problemas

### Fase 3: Benchmark Completo (4-8 horas)

**Objetivo**: Executar benchmark extensivo com múltiplas configurações

**Configuração**:
- n_bits: [6, 8, 10]
- n_codes: [16, 32, 64]
- k: [5, 10, 20]
- hamming_radius: [1, 2, 3]
- max_two_swap_iters: [10, 20, 50]
- num_tunneling_steps: [0, 5, 10]
- mode: ["two_swap_only", "full"]
- n_runs: 5

**Comando**:
```bash
python scripts/run_sprint8_benchmark.py \
    --dataset synthetic \
    --n-bits 6,8,10 \
    --n-codes 16,32,64 \
    --k 5,10,20 \
    --hamming-radius 1,2,3 \
    --max-iters 10,20,50 \
    --tunneling-steps 0,5,10 \
    --mode two_swap_only,full \
    --n-runs 5 \
    --output experiments/real/results_sprint8_complete.json
```

**Resultados Esperados**:
- Grid completo de resultados
- Análise de sensibilidade a parâmetros
- Identificação de configurações ótimas

### Fase 4: Análise e Relatório (1-2 horas)

1. **Análise de Resultados**:
   - Comparar GTH vs baselines
   - Identificar melhores configurações
   - Analisar trade-offs (recall vs tempo)

2. **Geração de Relatório**:
   - Tabelas comparativas
   - Gráficos de recall vs parâmetros
   - Análise estatística (significância)

3. **Documentação**:
   - Atualizar `RECALL_RESULTS_SUMMARY.md`
   - Criar `SPRINT8_BENCHMARK_RESULTS.md`
   - Documentar descobertas

## Estrutura de Resultados

### Formato JSON

```json
{
  "metadata": {
    "sprint": "8",
    "date": "2025-01-27",
    "dataset": "synthetic",
    "n_base": 1000,
    "n_queries": 100,
    "dim": 64
  },
  "baselines": {
    "hyperplane": {
      "radius_1": {
        "recall": 0.13,
        "search_time_ms": 2.5,
        "n_runs": 3,
        "std": 0.01
      },
      "radius_2": { ... }
    },
    "p_stable": { ... },
    "random_proj": { ... }
  },
  "gth_sprint8": {
    "hyperplane": {
      "config_1": {
        "n_bits": 6,
        "n_codes": 16,
        "k": 10,
        "radius": 1,
        "max_iters": 10,
        "tunneling_steps": 0,
        "recall": 0.08,
        "j_phi_cost": 2.3,
        "j_phi_improvement": 0.12,
        "build_time_s": 15.2,
        "search_time_ms": 3.1,
        "coverage": 0.15,
        "n_runs": 3,
        "std": 0.02
      },
      ...
    },
    "p_stable": { ... }
  },
  "comparisons": {
    "gth_vs_baseline": {
      "hyperplane": {
        "recall_improvement": -0.05,
        "relative_improvement": -38.5,
        "is_better": false
      }
    }
  }
}
```

## Critérios de Sucesso

### Mínimo Aceitável

1. ✅ **GTH não pior que baseline**: Recall GTH ≥ 0.9 × Baseline
2. ✅ **Melhoria após otimização**: J(φ) melhora ≥ 5%
3. ✅ **Cobertura razoável**: Hamming ball coverage ≥ 20% (radius=2)

### Objetivo Ideal

1. ✅ **GTH melhor que baseline**: Recall GTH > Baseline
2. ✅ **Melhoria significativa**: Recall GTH ≥ 1.1 × Baseline
3. ✅ **Cobertura alta**: Hamming ball coverage ≥ 50% (radius=2)

## Riscos e Mitigações

### Risco 1: Dados Reais Não Disponíveis
- **Mitigação**: Usar dataset sintético gerado

### Risco 2: Benchmark Muito Lento
- **Mitigação**: Começar com modo rápido, usar paralelização

### Risco 3: Recall Ainda Pior que Baseline
- **Mitigação**: Documentar problemas, planejar próximas melhorias

### Risco 4: Erros em Scripts
- **Mitigação**: Testar com configuração mínima primeiro

## Checklist de Execução

### Antes de Começar

- [ ] Verificar que dados estão disponíveis
- [ ] Validar que scripts funcionam com Sprint 8
- [ ] Criar script `run_sprint8_benchmark.py`
- [ ] Testar com configuração mínima

### Durante Execução

- [ ] Executar validação rápida primeiro
- [ ] Verificar resultados intermediários
- [ ] Executar benchmark completo
- [ ] Monitorar progresso e tempo

### Após Execução

- [ ] Validar que todos os resultados foram coletados
- [ ] Gerar análise e relatório
- [ ] Atualizar documentação
- [ ] Comparar com resultados anteriores

## Próximos Passos Imediatos

1. ✅ **Criar script `run_sprint8_benchmark.py`** - COMPLETO
2. ✅ **Criar script `analyze_sprint8_benchmark_results.py`** - COMPLETO
3. ⏳ **Testar com configuração mínima**
4. ⏳ **Executar validação rápida**
5. ⏳ **Analisar resultados iniciais**
6. ⏳ **Executar benchmark completo (se validação rápida for positiva)**

## Scripts Criados

### 1. `scripts/run_sprint8_benchmark.py` ✅

**Funcionalidades**:
- Carrega dados reais ou gera sintéticos
- Executa baselines (Hyperplane LSH, p-stable LSH)
- Executa GTH Sprint 8 com múltiplas configurações
- Coleta métricas detalhadas (recall, tempo, J(φ), cobertura)
- Gera comparações automáticas
- Salva resultados em JSON

**Uso**:
```bash
# Validação rápida
python scripts/run_sprint8_benchmark.py \
    --dataset synthetic \
    --n-bits 6,8 \
    --n-codes 16,32 \
    --k 10 \
    --hamming-radius 1,2 \
    --max-iters 10,20 \
    --tunneling-steps 0 \
    --mode two_swap_only \
    --n-runs 3 \
    --quick \
    --output experiments/real/results_sprint8_quick.json

# Benchmark completo
python scripts/run_sprint8_benchmark.py \
    --dataset synthetic \
    --n-bits 6,8,10 \
    --n-codes 16,32,64 \
    --k 5,10,20 \
    --hamming-radius 1,2,3 \
    --max-iters 10,20,50 \
    --tunneling-steps 0,5,10 \
    --mode two_swap_only,full \
    --n-runs 5 \
    --output experiments/real/results_sprint8_complete.json
```

### 2. `scripts/analyze_sprint8_benchmark_results.py` ✅

**Funcionalidades**:
- Carrega resultados JSON
- Gera tabelas comparativas
- Identifica melhores configurações
- Gera relatório em Markdown

**Uso**:
```bash
python scripts/analyze_sprint8_benchmark_results.py \
    --input experiments/real/results_sprint8_quick.json \
    --output experiments/real/SPRINT8_BENCHMARK_RESULTS.md
```

## Referências

- `experiments/real/RECALL_RESULTS_SUMMARY.md` - Resultados anteriores
- `experiments/real/SPRINT8_RESULTS_SUMMARY.md` - Mudanças da Sprint 8
- `scripts/benchmark_distribution_aware.py` - Script de benchmark existente
- `scripts/run_real_experiment_baseline.py` - Script de baseline
- `scripts/run_real_experiment_gray_tunneled.py` - Script de GTH (antigo)

