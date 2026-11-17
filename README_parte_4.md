# Parte 4 evoluída — Classificador de Área de Atendimento

Arquivo: `4_caixeiro_viajante.py`
Base: `renatoaloi/Curso-Python-IA-AloiTech` → `aulas/aula_13_treinamento_pytorch/4_caixeiro_viajante.py` (PythonIA #13).

> Obs. do professor: o script se chama "caixeiro viajante", mas na prática ele
> resolve um **problema de classificação geográfica**: decidir se uma coordenada
> está dentro da área de entrega. Mantive o nome original por fidelidade à aula.

## Como rodar

```powershell
.\venv\Scripts\python.exe 4_caixeiro_viajante.py            # treina, salva e testa
.\venv\Scripts\python.exe 4_caixeiro_viajante.py --recarregar # só carrega o .pt
.\venv\Scripts\python.exe 4_caixeiro_viajante.py --plot      # gera o gráfico (opcional)
```

Para o gráfico, instale `matplotlib numpy` no venv. Sem eles, o script avisa e
continua funcionando normalmente.

## Comparativo: original vs. evoluído

| Item | Original | Versão evoluída |
|---|---|---|
| Dados | 4 coordenadas manuais, com rótulos ambíguos | 800 pontos sintéticos, rótulo calculado pela distância real ao hub + 3% de ruído |
| Pré-processamento | min-max direto em lat/lon crua | conversão lat/lon → km (equiretangular), depois padronização com média/desvio do treino |
| Validação | nenhuma | split 80/20 com semente fixa |
| Treinamento | 1000 épocas fixas | até 1500 épocas com **early stopping** (escolhe a melhor época) |
| Métricas | só a perda | acurácia, precisão e recall |
| Reutilização | só dentro do script | `area_atendimento.pt` + classe `ClassificadorAreaAtendimento.prever(lat, lon)` |
| Visualização | estática, fora do código | `--plot` gera `graficos/area_atendimento.png` |
| Robustez | — | testes automáticos de 4 coordenadas conhecidas ao final |

## Resultado obtido no ambiente local

- Melhor época: 1021 (early stop na 1101)
- Validação: **acurácia 96,9% | precisão 93,5% | recall 90,6%**
- Testes de coordenadas: **4/4 corretos**

## Camada de API (próximo passo já entregue)

O modelo também está exposto via FastAPI em `api/main.py`, com testes em
`tests/test_api.py`. Veja o [README_API.md](README_API.md) para subir o servidor
e consumir `POST /atendimento`.

## Por que isso conta para uma vaga (ex.: iFood)

- Mostra que você entende **feature engineering geográfico**, não só `model.fit()`.
- Evita **vazamento de dados**: normalização calculada só no treino e salva junto com o modelo.
- Entrega **modelo versionado/serializado** e uma interface (`prever`) que pode virar
  um endpoint FastAPI depois.
- Inclui **validação e métricas**, que é o que separa um script de estudo de um
  projeto apresentável em entrevista.
