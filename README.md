# Modelo de Experiência Robótica — Classificador de Área de Atendimento

## Introdução

Este mini projeto é um **estudo prático de Inteligência Artificial com Python e PyTorch** que evolui passo a passo até virar um **serviço pronto para uso**: um modelo que decide se uma coordenada (latitude/longitude) está dentro da área de entrega de um hub, exposto via uma API REST.

A ideia não é só "treinar um modelo". O projeto mostra o caminho completo: começa nos conceitos básicos de tensores, passa por portas lógicas, chega ao tratamento de dados geográficos, salva o modelo treinado e termina com uma API validada e testada.

> **Sobre o nome:** o arquivo `4_caixeiro_viajante.py` mantém o nome original da aula, mas o problema resolvido na prática é de **classificação geográfica**, não um caixeiro viajante clássico.

## O que o projeto faz

O fluxo do estudo está dividido em quatro etapas:

| Etapa | Arquivo | O que acontece |
| --- | --- | --- |
| 1 | `1_tensores_basicos.py` | Mostra o que é um tensor (escalar, vetor, matriz, cubo), com formato, dimensões e tipo |
| 2 | `2_hello_word.py` | Treina um perceptron para aprender a porta lógica **OR** (linearmente separável) |
| 3 | `3_valores.py` | Usa uma rede com camada oculta para aprender a porta **XOR**, que exige não linearidade |
| 4 | `4_caixeiro_viajante.py` | Treina o classificador da área de atendimento e salva o modelo em `area_atendimento.pt` |

### A etapa principal (parte 4)

O classificador responde à pergunta: **"esta coordenada está dentro do raio de entrega?"**

Para isso ele:

1. Gera **800 pontos sintéticos** ao redor do hub (São Paulo: `-23.5505, -46.6333`), com um raio de entrega de **3 km** e 3% de ruído nos rótulos para simular erros de cadastro.
2. Converte latitude/longitude para **deslocamento em quilômetros** (aproximação equiretangular) — um tratamento geográfico real, em vez de usar lat/lon crus.
3. Normaliza os dados usando média e desvio calculados **apenas no treino**, evitando vazamento de dados.
4. Separa 80% para treino e 20% para validação com semente fixa (resultado reproduzível).
5. Treina uma rede `2 → 16 → 16 → 1` com **early stopping**, escolhendo a melhor época automaticamente.
6. Avalia com **acurácia, precisão e recall** e salva tudo (pesos, média, desvio, hub e raio) em `area_atendimento.pt`.
7. Disponibiliza a classe `ClassificadorAreaAtendimento.prever(lat, lon)` para reutilizar o modelo sem treinar de novo.

Resultado obtido no ambiente local: **96,9% de acurácia**, 93,5% de precisão e 90,6% de recall, com 4/4 coordenadas de teste corretas.

### A camada de API

O mesmo modelo é exposto por uma **API FastAPI** em `api/main.py`, com documentação automática e validação de entrada:

| Método | Rota | Função |
| --- | --- | --- |
| `GET` | `/` | Mensagem de boas-vindas |
| `GET` | `/health` | Verificação de saúde |
| `GET` | `/info` | Época, perda, hub e raio do modelo |
| `POST` | `/atendimento` | Classifica uma coordenada |
| `POST` | `/atendimento/lote` | Classifica de 1 a 1000 coordenadas de uma vez |

A validação com Pydantic garante latitude entre -90 e 90, longitude entre -180 e 180 e lote com 1 a 1000 itens. Entradas inválidas retornam `422`.

Os testes em `tests/test_api.py` cobrem os endpoints e as validações: **9 testes passando**.

## Estrutura do repositório

```text
1_tensores_basicos.py       # conceito de tensor
2_hello_word.py             # perceptron OR
3_valores.py                # rede neural XOR
4_caixeiro_viajante.py      # treino e salvamento do classificador
area_atendimento.pt         # modelo serializado
graficos/                   # visualização da fronteira de decisão
api/                        # camada HTTP (FastAPI + Pydantic)
tests/                      # testes automatizados
requirements.txt            # dependências
README_parte_4.md           # detalhes do treino e comparação original vs evoluído
README_API.md               # detalhes dos endpoints e exemplos de chamada
README.md                   # esta introdução geral
```

## Como executar

```powershell
# 1. Instalar as dependências
pip install -r requirements.txt

# 2. Treinar o modelo (gera area_atendimento.pt)
python 4_caixeiro_viajante.py

# 3. Subir a API
python -m uvicorn api.main:app --reload

# 4. Rodar os testes
python -m pytest -q
```

Depois de subir a API, acesse a documentação interativa em <http://127.0.0.1:8000/docs>.

## Tecnologias usadas

- **Python** e **PyTorch** — treinamento e inferência do modelo
- **FastAPI** e **Pydantic** — API REST com validação
- **Uvicorn** — servidor HTTP
- **Pytest** — testes automatizados
- **NumPy/Matplotlib** — gráfico opcional da fronteira de decisão

## Documentação detalhada

- [README_parte_4.md](README_parte_4.md) — explicação completa do classificador e do treino
- [README_API.md](README_API.md) — endpoints, exemplos de requisição e testes da API
