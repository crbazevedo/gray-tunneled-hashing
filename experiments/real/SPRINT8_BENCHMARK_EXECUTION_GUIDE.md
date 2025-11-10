# Sprint 8 - Guia de Execução do Benchmark

**Data**: 2025-01-27  
**Status**: Scripts Criados - Pronto para Execução

## Scripts Disponíveis

### 1. `scripts/run_sprint8_benchmark.py` ✅

Script principal para executar o benchmark completo.

**Funcionalidades**:
- Carrega dados reais ou gera sintéticos
- Executa baselines (Hyperplane LSH, p-stable LSH)
- Executa GTH Sprint 8 com múltiplas configurações
- Coleta métricas detalhadas
- Gera comparações automáticas
- Salva resultados em JSON

### 2. `scripts/analyze_sprint8_benchmark_results.py` ✅

Script para análise e geração de relatórios.

**Funcionalidades**:
- Carrega resultados JSON
- Gera tabelas comparativas
- Identifica melhores configurações
- Gera relatório em Markdown

## Execução Passo a Passo

### Passo 1: Validação Rápida (Recomendado Primeiro)

**Objetivo**: Validar que tudo funciona e obter resultados iniciais

**Comando**:
```bash
cd /Users/59388/coding/gray-tunneled-hashing

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

**Tempo estimado**: 30-60 minutos

**Resultados esperados**:
- Arquivo JSON com resultados
- Comparação GTH vs baselines
- Validação de que recall melhorou (ou não)

### Passo 2: Análise dos Resultados Rápidos

**Comando**:
```bash
python scripts/analyze_sprint8_benchmark_results.py \
    --input experiments/real/results_sprint8_quick.json \
    --output experiments/real/SPRINT8_BENCHMARK_QUICK_RESULTS.md
```

**Saída**:
- Tabelas comparativas no terminal
- Relatório Markdown salvo

### Passo 3: Benchmark Completo (Se Validação Rápida for Positiva)

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

**Tempo estimado**: 4-8 horas

**Resultados esperados**:
- Grid completo de resultados
- Análise de sensibilidade a parâmetros
- Identificação de configurações ótimas

### Passo 4: Análise dos Resultados Completos

**Comando**:
```bash
python scripts/analyze_sprint8_benchmark_results.py \
    --input experiments/real/results_sprint8_complete.json \
    --output experiments/real/SPRINT8_BENCHMARK_COMPLETE_RESULTS.md
```

## Configurações Recomendadas

### Validação Rápida (Quick Mode)

```yaml
n_bits: [6, 8]
n_codes: [16, 32]
k: [10]
hamming_radius: [1, 2]
max_two_swap_iters: [10, 20]
num_tunneling_steps: [0]
mode: ["two_swap_only"]
n_runs: 3
```

**Total de configurações**: ~16 GTH + 4 baselines = 20

### Benchmark Completo

```yaml
n_bits: [6, 8, 10]
n_codes: [16, 32, 64]
k: [5, 10, 20]
hamming_radius: [1, 2, 3]
max_two_swap_iters: [10, 20, 50]
num_tunneling_steps: [0, 5, 10]
mode: ["two_swap_only", "full"]
n_runs: 5
```

**Total de configurações**: ~3,240 GTH + 18 baselines = 3,258

## Interpretação dos Resultados

### Métricas Coletadas

1. **Recall@k**: Fração de neighbors verdadeiros recuperados
   - **Objetivo**: GTH > Baseline
   - **Mínimo aceitável**: GTH ≥ 0.9 × Baseline

2. **Build Time**: Tempo de construção do índice
   - **Esperado**: GTH > Baseline (GTH tem otimização)
   - **Aceitável**: < 60s para N=1000

3. **Search Time**: Tempo de busca
   - **Esperado**: Similar ao baseline
   - **Aceitável**: < 10ms por query

4. **J(φ) Improvement**: Melhoria do objetivo
   - **Esperado**: > 0% (melhoria)
   - **Ideal**: > 10%

5. **Hamming Ball Coverage**: % de neighbors no ball
   - **Esperado**: > 20% (radius=2)
   - **Ideal**: > 50% (radius=2)

### Comparações

O script gera automaticamente comparações entre GTH e baselines:

- **recall_improvement**: Diferença absoluta (GTH - Baseline)
- **relative_improvement_pct**: Diferença relativa em %
- **is_better**: Boolean indicando se GTH é melhor

## Troubleshooting

### Erro: "Dataset not found"
- **Solução**: Use `--dataset synthetic` para gerar dados sintéticos

### Erro: "n_codes > 2**n_bits"
- **Solução**: Ajuste n_codes para ser ≤ 2**n_bits

### Erro: "Memory error"
- **Solução**: Reduza tamanho do dataset ou use modo rápido

### Benchmark muito lento
- **Solução**: Use `--quick` ou reduza configurações

## Próximos Passos Após Execução

1. **Analisar resultados** usando `analyze_sprint8_benchmark_results.py`
2. **Comparar com resultados anteriores** (Sprint 5, 6, 7)
3. **Identificar melhores configurações**
4. **Documentar descobertas** em `SPRINT8_BENCHMARK_RESULTS.md`
5. **Atualizar** `RECALL_RESULTS_SUMMARY.md` com novos resultados

## Referências

- `experiments/real/SPRINT8_BENCHMARK_PLAN.md` - Plano detalhado
- `experiments/real/RECALL_RESULTS_SUMMARY.md` - Resultados anteriores
- `scripts/run_sprint8_benchmark.py` - Script de benchmark
- `scripts/analyze_sprint8_benchmark_results.py` - Script de análise

