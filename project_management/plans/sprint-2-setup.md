# Sprint 2: Dataset Setup

## Dataset Escolhido

**Dataset**: Quora Question Pairs (ou similar dataset de texto)

**Estratégia**: 
- Usar sentence-transformers para gerar embeddings de texto
- Trabalhar com embeddings já pré-computados em formato `.npy` para evitar data wrangling pesado na sprint
- Focar em iterar rápido sobre o algoritmo, não na pipeline de embeddings

## Formato dos Dados

### Estrutura de Arquivos

```
experiments/real/data/
├── quora_base_embeddings.npy          # Embeddings do corpus base (N, dim)
├── quora_queries_embeddings.npy       # Embeddings das queries (Q, dim)
└── quora_ground_truth_indices.npy     # Ground truth kNN (Q, k) - gerado pelo script
```

### Especificações

- **Base embeddings**: Shape `(N, dim)` onde `N` é o número de documentos no corpus
- **Queries embeddings**: Shape `(Q, dim)` onde `Q` é o número de queries
- **Ground truth**: Shape `(Q, k)` onde `ground_truth[i, j]` é o índice do j-ésimo vizinho mais próximo da query `i` no corpus base

## Geração dos Embeddings

### Processo (para referência, não parte da sprint)

1. Baixar dataset de texto (ex: Quora Question Pairs)
2. Usar sentence-transformers (ex: `all-MiniLM-L6-v2`) para gerar embeddings
3. Separar em corpus base e queries
4. Salvar como `.npy` files

### Exemplo de Código (não executado na sprint)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
# ... processar textos ...
base_embeddings = model.encode(base_texts)
queries_embeddings = model.encode(query_texts)
np.save('quora_base_embeddings.npy', base_embeddings)
np.save('quora_queries_embeddings.npy', queries_embeddings)
```

## Normalização

- Embeddings podem ser normalizados (L2) se necessário
- Loader deve garantir consistência de formato
- Não há normalização forçada por padrão (dados já vêm prontos)

## Dataset Alternativo

Se Quora Question Pairs não estiver disponível, usar:
- **MS MARCO** (pequeno subset)
- **SQuAD** (sentences)
- Ou qualquer dataset de texto com embeddings pré-computados

## Considerações

- Dataset deve ter pelo menos 1000 documentos base para ser interessante
- Queries devem ser pelo menos 100 para estatísticas significativas
- Dimensão dos embeddings: 384 (MiniLM) ou 768 (BERT-base) são razoáveis

