"""Rede neural com camada oculta para aprender a porta lógica XOR.

Atenção: a tabela de respostas aqui é XOR (0, 1, 1, 0), não OR.
A camada oculta é necessária porque o XOR não é separável linearmente.
"""

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

# 1. Dados de treinamento (porta XOR).
entradas = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    dtype=torch.float32,
)
saidas = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)

# 2. Modelo: 2 entradas -> 4 neurônios ocultos -> 1 saída.
modelo = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid(),
)

# 3. Critério de erro e otimizador.
criterio = nn.BCELoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.05)

# 4. Loop de treinamento.
print("Treinando a IA para aprender a porta XOR...\n")
for epoca in range(1, 5001):
    previsao = modelo(entradas)
    erro = criterio(previsao, saidas)

    otimizador.zero_grad()
    erro.backward()
    otimizador.step()

    if epoca % 500 == 0:
        print(f"Época {epoca:4d}/5000 | erro: {erro.item():.4f}")

# Pontos fora da faixa 0/1 mostram como a rede extrapola o que aprendeu.
testes = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 2.0], [-2.0, -2.0]],
    dtype=torch.float32,
)

with torch.no_grad():
    previsoes_treino = modelo(entradas)
    previsoes_teste = modelo(testes)
    acertos = (previsoes_treino >= 0.5).float().eq(saidas).float()
    acuracia = acertos.mean().item()

print(f"\nAcurácia nos dados de treino: {acuracia:.0%}")
print("\nTeste final:")
for entrada, previsao in zip(testes, previsoes_teste):
    valor = previsao.item()
    classe = "TRUE" if valor >= 0.5 else "FALSE"
    print(f"Entrada: {entrada.tolist()} -> previsão: {valor:.4f} ({classe})")
