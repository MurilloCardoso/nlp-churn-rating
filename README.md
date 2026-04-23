## Contexto

Uma operadora de telecomunicações está perdendo clientes em ritmo
acelerado. A diretoria precisa de um modelo preditivo de churn que classifique
clientes com risco de cancelamento. Assim, o grupo deve construir o projeto do
zero ao modelo servido via API, aplicando todas as boas práticas de engenharia
de ML aprendidas na Fase 1.

O modelo central de entrega é uma rede neural (MLP), treinada com
PyTorch, comparada com baselines (Scikit-Learn) e rastreada com MLflow.

## Setup

### Pré-requisitos

- [uv](https://docs.astral.sh/uv/getting-started/installation/) instalado

### Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/nlp-churn-rating.git
cd nlp-churn-rating

# Cria o venv e instala as dependências
uv sync

# Para incluir dependências de desenvolvimento (pytest, etc.)
uv sync --extra dev
```

### Usando os notebooks

No VS Code, selecione o interpretador `.venv/bin/python` como kernel do Jupyter. O `ipykernel` já está incluso nas dependências.

### Executando os testes

```bash
uv run pytest
```

---





Num terminal, sobe a API:


uvicorn src.api:app --reload



