# Sprint 5.1 - Validação e Experimentos

## Objetivo

Completar validações e experimentos faltantes da Sprint 5, garantindo que todas as hipóteses sejam validadas empiricamente antes de prosseguir para Sprint 6.

## Duração Estimada

2-3 dias

## Tarefas

### 1. Melhorar Teste de Preservação de Colisões (Prioridade ALTA)

**Status Atual**: ✅ Testes passam (4/4), mas validação de propriedade incompleta.

**Problema**: Teste `test_lsh_collision_preservation()` existe mas não valida corretamente a propriedade de preservação de colisões.

**Tarefas**:
- [x] Bug corrigido: `encoder=None` quando `lsh_family` fornecido
- [x] Testes passam (4/4)
- [ ] **FALTA**: Implementar validação correta de preservação de colisões:
  - Hash embeddings antes de GTH: `codes_before = lsh.hash(embeddings)`
  - Aplicar GTH permutation: `codes_after = apply_permutation(codes_before, ...)`
  - Verificar: se `codes_before[i] == codes_before[j]`, então `codes_after[i] == codes_after[j]`
  - Medir % de colisões preservadas (deve ser 100%)
- [ ] Documentar resultados da validação

**Critério de Aceite**: 
- Teste valida que 100% das colisões são preservadas
- Resultado documentado com evidência empírica

### 2. Completar Script de Benchmark (Prioridade ALTA)

**Problema**: Script `benchmark_lsh_vs_random_proj.py` tem implementação incompleta (recall sempre 0.0).

**Tarefas**:
- [ ] Corrigir cálculo de recall@k:
  - Mapear candidatos do Hamming ball para índices reais do dataset
  - Calcular recall@k corretamente usando `recall_at_k()`
- [ ] Adicionar suporte para múltiplos runs (média, std)
- [ ] Adicionar métricas de latência (build time, search time)
- [ ] Testar script com configuração mínima

**Critério de Aceite**: Script produz resultados confiáveis com recall@k calculado corretamente.

### 3. Executar Experimentos Empíricos (Prioridade CRÍTICA)

**Problema**: ❌ **NENHUM EXPERIMENTO FOI EXECUTADO** - Sprint 5 não terminou com experimento validando hipóteses.

**Tarefas**:
- [ ] Executar `benchmark_lsh_vs_random_proj.py` com configuração mínima:
  - n_bits=8, n_codes=32, n_samples=100, n_queries=20, k=5
- [ ] Executar com diferentes Hamming ball radius (0, 1, 2)
- [ ] Executar para todos os métodos:
  - `baseline_hyperplane`, `baseline_p_stable`, `baseline_random_proj`
  - `hyperplane`, `p_stable`, `random_proj` (com GTH)
- [ ] Gerar resultados em JSON
- [ ] Analisar resultados e criar tabelas comparativas

**Critério de Aceite**: 
- Pelo menos 3 experimentos executados com sucesso
- Resultados salvos em `experiments/real/results_sprint5.json`
- Tabelas comparativas geradas em `results_sprint5.md`

### 4. Validar Hipóteses Empiricamente (Prioridade ALTA)

**Hipóteses a Validar**:

- **H2**: GTH preserva garantias de colisão LSH
  - Métrica: % de colisões preservadas (deve ser 100%)
  
- **H3**: Hamming ball expansion melhora recall@k
  - Métrica: recall@k para radius=0 vs. radius=1 vs. radius=2
  - Esperado: recall aumenta com radius
  
- **H4**: GTH melhora recall@k para LSH e random projection
  - Métrica: recall@k baseline vs. recall@k com GTH
  - Esperado: GTH melhora recall em ambos os casos
  
- **H5**: LSH + GTH vs. random projection + GTH
  - Métrica: recall@k comparativo
  - Esperado: LSH pode ter vantagem dependendo do dataset

**Critério de Aceite**: Todas as hipóteses validadas com evidência empírica.

### 5. Documentar Resultados (Prioridade ALTA)

**Tarefas**:
- [ ] Criar `experiments/real/results_sprint5.md` com:
  - Tabelas comparativas de recall@k
  - Análise de impacto do Hamming ball radius
  - Validação de preservação de colisões
  - Conclusões e recomendações
- [ ] Criar seção no sprint-log.md documentando resultados

**Critério de Aceite**: Documentação completa com resultados e análises.

### 6. Melhorar Script de Benchmark (Prioridade MÉDIA)

**Tarefas**:
- [ ] Adicionar suporte para calcular recall@k corretamente
- [ ] Adicionar métricas de latência
- [ ] Adicionar suporte para múltiplos runs (média, std)
- [ ] Adicionar opção para salvar resultados em CSV

**Critério de Aceite**: Script produz resultados confiáveis e completos.

## Entregáveis

1. ✅ Testes de integração corrigidos e passando
2. ✅ Validação de preservação de colisões implementada e documentada
3. ✅ Experimentos executados com resultados em JSON
4. ✅ Análise de resultados em `results_sprint5.md`
5. ✅ Todas as hipóteses validadas empiricamente
6. ✅ Documentação completa no sprint-log

## Critérios de Aceite Finais

- [ ] Todos os testes passam (100% success rate)
- [ ] Preservação de colisões validada (100% preservação)
- [ ] Pelo menos 3 experimentos executados com sucesso
- [ ] Resultados documentados com tabelas e análises
- [ ] Hipóteses H2, H3, H4, H5 validadas com evidência
- [ ] Sprint 5.1 documentada no sprint-log.md

## Notas

- Focar em validação empírica antes de otimizações
- Usar configurações pequenas para testes rápidos
- Documentar limitações e próximos passos

