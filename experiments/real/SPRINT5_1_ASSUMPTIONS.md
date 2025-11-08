# Sprint 5.1 - Assumptions Explícitas

Este documento lista todas as assumptions (suposições) explícitas usadas na Sprint 5.1, junto com seus planos de validação.

## A1: LSH Properties

**Assumption**: LSH families têm propriedades teóricas corretas de colisão.

**Detalhes**:
- **Hyperplane LSH**: Para cada bit, a probabilidade de colisão é aproximadamente `1 - arccos(sim)/π`, onde `sim` é a similaridade de cosseno entre vetores.
- **p-stable LSH**: Para cada bit, a probabilidade de colisão é aproximadamente `1/(1 + d/w)`, onde `d` é a distância L2 e `w` é o parâmetro de largura.

**Validação**:
- Testes unitários em `tests/test_lsh_families.py` validam `collision_probability()` e `validate_lsh_properties()`.
- Função `validate_lsh_properties()` verifica empiricamente que as propriedades teóricas são satisfeitas.

**Status**: ✅ Validado (10/10 testes passando)

---

## A2: GTH Preserves Collisions

**Assumption**: GTH preserva bucket membership (colisões LSH).

**Detalhes**:
- Se dois embeddings têm o mesmo código LSH antes de GTH (`c_i == c_j`), eles devem ter o mesmo código depois de GTH (`σ(c_i) == σ(c_j)`).
- GTH apenas relabela códigos, não altera bucket membership.
- Esta é uma propriedade fundamental: GTH preserva as garantias teóricas de LSH.

**Validação**:
- Teste `test_validate_collision_preservation_100_percent()` em `tests/test_collision_validation.py`.
- Função `validate_collision_preservation()` em `experiments/collision_validation.py` verifica que 100% das colisões são preservadas.

**Esperado**: 100% de preservação (exatamente, não aproximadamente).

**Status**: ⚠️ Implementado, aguardando validação empírica

---

## A3: Hamming Ball Improves Recall

**Assumption**: Hamming ball expansion melhora recall@k.

**Detalhes**:
- Maior radius → mais candidatos → maior recall@k.
- Trade-off: maior radius → mais latência (mais candidatos para processar).

**Validação**:
- Experimento 1: Comparar recall@k para radius=0, 1, 2.
- Hipótese H3: `recall(radius=2) > recall(radius=1) > recall(radius=0)`.

**Status**: ⚠️ Aguardando experimentos

---

## A4: GTH Improves Recall

**Assumption**: GTH melhora recall@k comparado a baseline.

**Detalhes**:
- GTH alinha Hamming distance com semantic distance.
- Deve melhorar recall@k comparado a métodos baseline (sem GTH).

**Validação**:
- Comparação baseline vs. GTH em Experimento 2.
- Hipótese H4: `recall_gth > recall_baseline` (estatisticamente significativo).

**Status**: ⚠️ Aguardando experimentos

---

## A5: Dataset Synthetic is Representative

**Assumption**: Embeddings sintéticos (Gaussian) são representativos para validação inicial.

**Detalhes**:
- Embeddings sintéticos gerados com `np.random.randn()` são usados para validação inicial.
- Resultados podem variar em datasets reais.
- Esta é uma limitação conhecida: validação em dados sintéticos não garante performance em dados reais.

**Validação**:
- Documentar limitações explicitamente.
- Nota: Para validação final, usar datasets reais (Sprint 6+).

**Status**: ✅ Documentado

---

## Resumo de Validação

| Assumption | Status | Métrica | Esperado |
|------------|--------|---------|----------|
| A1: LSH Properties | ✅ Validado | Testes unitários | 10/10 passando |
| A2: GTH Preserves Collisions | ⚠️ Implementado | Preservation rate | 100% |
| A3: Hamming Ball Improves Recall | ⚠️ Aguardando | recall@k vs. radius | Monotonicamente crescente |
| A4: GTH Improves Recall | ⚠️ Aguardando | Improvement over baseline | > 0 (significativo) |
| A5: Synthetic Dataset Representative | ✅ Documentado | Limitação conhecida | N/A |

---

## Notas

- **A2** é crítica: se não for 100%, há um bug na implementação.
- **A3** e **A4** são hipóteses empíricas que precisam ser validadas com experimentos.
- **A5** é uma limitação conhecida que será endereçada em sprints futuras.

