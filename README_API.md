# API de Área de Atendimento (FastAPI)

Camada HTTP sobre o modelo treinado em `4_caixeiro_viajante.py`. É o mesmo
classificador da parte 4, agora consumível por qualquer cliente (app, site,
outro microsserviço).

## Estrutura

```
api/
  __init__.py     # pacote
  schemas.py      # validação Pydantic dos request/response
  main.py         # endpoints + carregamento do modelo .pt
tests/
  test_api.py     # 9 testes com TestClient
area_atendimento.pt
```

## Subir o servidor

```powershell
.\venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Depois abra:

- Documentação interativa: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health
- Info do modelo: http://127.0.0.1:8000/info

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/` | mensagem de boas-vindas |
| `GET` | `/health` | health check |
| `GET` | `/info` | época, perda, hub e raio do modelo |
| `POST` | `/atendimento` | classifica 1 coordenada |
| `POST` | `/atendimento/lote` | classifica até 1000 coordenadas de uma vez |

### Exemplo — uma coordenada

```powershell
curl -X POST http://127.0.0.1:8000/atendimento `
  -H "Content-Type: application/json" `
  -d '{"latitude": -23.5505, "longitude": -46.6333}'
```

Resposta:

```json
{
  "latitude": -23.5505,
  "longitude": -46.6333,
  "atendido": true,
  "probabilidade": 0.9613,
  "raio_km": 3.0
}
```

### Exemplo — lote

```powershell
curl -X POST http://127.0.0.1:8000/atendimento/lote `
  -H "Content-Type: application/json" `
  -d '{"coordenadas": [{"latitude": -23.5505, "longitude": -46.6333}, {"latitude": -23.5505, "longitude": -46.5843}]}'
```

Validações automáticas do Pydantic: latitude entre -90 e 90, longitude entre
-180 e 180, e lote com 1 a 1000 itens. Fora disso a API responde `422`.

## Testes

```powershell
.\venv\Scripts\python.exe -m pytest -q
```

Resultado local: **9 passed**.

## Por que isso valoriza o projeto

- Transforma o modelo de "script que roda no terminal" em **serviço**.
- Mostra **validação de entrada**, **serialização** e **tratamento de erro (422)**.
- Separa responsabilidades: treino (`4_caixeiro_viajante.py`) ≠ entrega (`api/`).
- Tem **testes automatizados**, que é o que normalmente cai em entrevista Backend/ML.
