"""Treina um perceptron simples para aprender a porta lógica OR."""

import torch
import torch.nn as nn
import torch.optim as optim

# Semente fixa para o resultado ser reproduzível a cada execução.
torch.manual_seed(42)

# 1. Dados de treinamento: entradas -> saída da porta OR.
entradas = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    dtype=torch.float32,
)
saidas = torch.tensor([[0.0], [1.0], [1.0], [1.0]], dtype=torch.float32)

# 2. Modelo: 2 entradas -> 1 saída, com sigmoid para ficar entre 0 e 1.
modelo = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

# 3. BCELoss é a função de erro adequada para classificação binária com sigmoid.
criterio = nn.BCELoss()
otimizador = optim.SGD(modelo.parameters(), lr=0.1)

# 4. Treinamento.
print("Treinando a IA para aprender a porta OR...\n")
for epoca in range(1, 1001):
    previsao = modelo(entradas)
    erro = criterio(previsao, saidas)

    otimizador.zero_grad()  # limpa os gradientes da época anterior
    erro.backward()         # calcula para onde ajustar os pesos
    otimizador.step()       # ajusta os pesos

    if epoca % 100 == 0:
        print(f"Época {epoca:4d}/1000 | erro: {erro.item():.4f}")

# 5. Teste em modo inferência (sem calcular gradientes).
testes = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
    dtype=torch.float32,
)
with torch.no_grad():
    previsoes = modelo(testes)

print("\nTeste final:")
for entrada, previsao in zip(testes, previsoes):
    valor = previsao.item()
    classe = "TRUE" if valor >= 0.5 else "FALSE"
    print(f"Entrada: {entrada.tolist()} -> previsão: {valor:.4f} ({classe})")
