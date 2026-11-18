 
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

# --- Configuração da "loja" -------------------------------------------------
HUB_LAT = -23.5505          # centro de São Paulo
HUB_LON = -46.6333
RAIO_KM = 3.0               # raio máximo de entrega
SEMENTE = 42
CAMINHO_MODELO = Path(__file__).with_name("area_atendimento.pt")
CAMINHO_GRAFICO = Path(__file__).with_name("graficos") / "area_atendimento.png"

KM_POR_GRAU_LAT = 111.32


# --- Geografia -> km --------------------------------------------------------
def km_por_grau_lon(lat: float) -> float:
    """Distância de 1 grau de longitude na latitude informada."""
    return KM_POR_GRAU_LAT * math.cos(math.radians(lat))


def lat_lon_para_km(lat, lon, hub_lat: float = HUB_LAT, hub_lon: float = HUB_LON):
    """Converte (lat, lon) para deslocamento em km (leste, norte) a partir do hub.

    Aceita números ou tensores. A aproximação equiretangular é excelente para
    distâncias curtas como uma região de delivery.
    """
    lat = torch.as_tensor(lat, dtype=torch.float32)
    lon = torch.as_tensor(lon, dtype=torch.float32)
    leste = (lon - hub_lon) * km_por_grau_lon(hub_lat)
    norte = (lat - hub_lat) * KM_POR_GRAU_LAT
    return torch.stack((leste, norte), dim=-1)


def km_para_lat_lon(km, hub_lat: float = HUB_LAT, hub_lon: float = HUB_LON):
    """Conversão inversa: (leste, norte) em km -> (lat, lon)."""
    leste, norte = torch.as_tensor(km, dtype=torch.float32).unbind(dim=-1)
    lat = hub_lat + norte / KM_POR_GRAU_LAT
    lon = hub_lon + leste / km_por_grau_lon(hub_lat)
    return lat, lon


# --- Dados ------------------------------------------------------------------
def gerar_dataset(quantidade: int = 800, ruido: float = 0.03, semente: int = SEMENTE):
    """Cria pontos numa caixa de 12 km e rotula pela distância real ao hub.

    Uma pequena fração dos rótulos é invertida para simular erros de cadastro,
    o que evita que a rede simplesmente decore os exemplos.
    """
    gerador = torch.Generator().manual_seed(semente)
    km = (torch.rand(quantidade, 2, generator=gerador) - 0.5) * 12.0
    distancias = km.norm(dim=1)
    rotulos = (distancias < RAIO_KM).float().unsqueeze(1)

    mascara_ruido = torch.rand(quantidade, generator=gerador) < ruido
    rotulos[mascara_ruido] = 1.0 - rotulos[mascara_ruido]
    return km, rotulos


def dividir_treino_validacao(km, rotulos, fracao_validacao: float = 0.2):
    """Embaralha com semente fixa e separa 20% para validação."""
    gerador = torch.Generator().manual_seed(SEMENTE)
    ordem = torch.randperm(len(km), generator=gerador)
    tamanho_val = int(len(km) * fracao_validacao)
    return (
        km[ordem[tamanho_val:]],
        rotulos[ordem[tamanho_val:]],
        km[ordem[:tamanho_val]],
        rotulos[ordem[:tamanho_val]],
    )


# --- Modelo -----------------------------------------------------------------
def construir_modelo() -> nn.Module:
    """Rede 2 -> 16 -> 16 -> 1 com sigmoid na saída."""
    return nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid(),
    )


def metricas(modelo, km_normalizado, rotulos):
    """Retorna acurácia, precisão e recall para a classe 'atendido'."""
    modelo.eval()
    with torch.no_grad():
        probabilidades = modelo(km_normalizado)
        previstos = (probabilidades >= 0.5).float()

        acuracia = previstos.eq(rotulos).float().mean().item()
        verdadeiros_positivos = (previstos * rotulos).sum().item()
        falsos_positivos = (previstos * (1 - rotulos)).sum().item()
        falsos_negativos = ((1 - previstos) * rotulos).sum().item()

    denominador_precisao = verdadeiros_positivos + falsos_positivos
    denominador_recall = verdadeiros_positivos + falsos_negativos
    precisao = verdadeiros_positivos / denominador_precisao if denominador_precisao else 0.0
    recall = verdadeiros_positivos / denominador_recall if denominador_recall else 0.0
    return acuracia, precisao, recall


def treinar(epocas: int = 1500, paciencia: int = 80) -> None:
    """Treina, escolhe a melhor época pela validação e salva o modelo."""
    torch.manual_seed(SEMENTE)

    km, rotulos = gerar_dataset()
    km_treino, y_treino, km_val, y_val = dividir_treino_validacao(km, rotulos)

    # Normalização com estatísticas do treino (não do dataset inteiro).
    media = km_treino.mean(dim=0)
    desvio = km_treino.std(dim=0).clamp_min(1e-6)
    treino_n = (km_treino - media) / desvio
    validacao_n = (km_val - media) / desvio

    modelo = construir_modelo()
    criterio = nn.BCELoss()
    otimizador = optim.Adam(modelo.parameters(), lr=0.01)

    melhor_perda = float("inf")
    melhor_estado = None
    melhor_epoca = 0
    esperas = 0

    print("Treinando a IA para reconhecer nossa área de atendimento...")
    for epoca in range(1, epocas + 1):
        modelo.train()
        previsao = modelo(treino_n)
        perda = criterio(previsao, y_treino)

        otimizador.zero_grad()
        perda.backward()
        otimizador.step()

        modelo.eval()
        with torch.no_grad():
            perda_val = criterio(modelo(validacao_n), y_val).item()

        if perda_val < melhor_perda - 1e-5:
            melhor_perda = perda_val
            melhor_estado = {chave: valor.detach().clone() for chave, valor in modelo.state_dict().items()}
            melhor_epoca = epoca
            esperas = 0
        else:
            esperas += 1
            if esperas >= paciencia:
                print(f"Early stop na época {epoca} (sem melhora desde a {melhor_epoca}).")
                break

        if epoca % 200 == 0:
            acuracia, _, _ = metricas(modelo, validacao_n, y_val)
            print(
                f"Época {epoca:4d}/{epocas} | perda treino {perda.item():.4f} "
                f"| perda val {perda_val:.4f} | acurácia val {acuracia:.1%}"
            )

    modelo.load_state_dict(melhor_estado)
    modelo.eval()
    torch.save(
        {
            "estado_modelo": modelo.state_dict(),
            "media": media,
            "desvio": desvio,
            "hub_lat": HUB_LAT,
            "hub_lon": HUB_LON,
            "raio_km": RAIO_KM,
            "epoca": melhor_epoca,
            "perda_validacao": melhor_perda,
        },
        CAMINHO_MODELO,
    )

    acuracia, precisao, recall = metricas(modelo, validacao_n, y_val)
    print(f"\nMelhor época: {melhor_epoca} | perda de validação: {melhor_perda:.4f}")
    print(f"Validação -> acurácia {acuracia:.1%} | precisão {precisao:.1%} | recall {recall:.1%}")
    print(f"Modelo salvo em: {CAMINHO_MODELO}")


# --- Reutilização do modelo -------------------------------------------------
class ClassificadorAreaAtendimento:
    """Carrega o modelo salvo e classifica coordenadas sem treinar de novo."""

    def __init__(self, caminho: Path = CAMINHO_MODELO):
        dados = torch.load(caminho, map_location="cpu", weights_only=True)
        self.hub_lat = dados["hub_lat"]
        self.hub_lon = dados["hub_lon"]
        self.raio_km = dados["raio_km"]
        self.epoca = dados["epoca"]
        self.perda_validacao = dados.get("perda_validacao")
        self.media = dados["media"]
        self.desvio = dados["desvio"]

        self.modelo = construir_modelo()
        self.modelo.load_state_dict(dados["estado_modelo"])
        self.modelo.eval()

    def prever_km(self, km):
        """Probabilidade de atendimento para pontos em km (leste, norte)."""
        km = torch.as_tensor(km, dtype=torch.float32)
        if km.ndim == 1:
            km = km.unsqueeze(0)
        with torch.no_grad():
            return self.modelo((km - self.media) / self.desvio).squeeze(-1)

    def prever(self, lat: float, lon: float):
        """Retorna (probabilidade, atendido) para uma latitude/longitude."""
        km = lat_lon_para_km(lat, lon, self.hub_lat, self.hub_lon)
        probabilidade = self.prever_km(km)
        return probabilidade, probabilidade >= 0.5


def ponto_a_partir_do_hub(leste_km: float, norte_km: float):
    """Monta uma (lat, lon) de teste a partir de deslocamentos em km."""
    lat, lon = km_para_lat_lon(
        torch.tensor([leste_km, norte_km]), HUB_LAT, HUB_LON
    )
    return lat.item(), lon.item()


def validar_classificador(classificador: ClassificadorAreaAtendimento) -> bool:
    """Testes rápidos com pontos claramente dentro/fora do raio."""
    lat_1km, lon_1km = ponto_a_partir_do_hub(1.0, 0.0)
    lat_5km, lon_5km = ponto_a_partir_do_hub(5.0, 0.0)
    lat_2km, lon_2km = ponto_a_partir_do_hub(-1.5, 1.5)

    casos = [
        ("Centro da loja", HUB_LAT, HUB_LON, True),
        ("1 km a leste", lat_1km, lon_1km, True),
        ("5 km a leste", lat_5km, lon_5km, False),
        ("~2 km a noroeste", lat_2km, lon_2km, True),
    ]

    print("\n--- TESTANDO NOVAS COORDENADAS ---")
    acertos = 0
    for nome, lat, lon, esperado in casos:
        probabilidade, atendido = classificador.prever(lat, lon)
        conf = probabilidade.item()
        ok = bool(atendido.item()) == esperado
        acertos += ok
        status = "✅ ATENDIDO" if atendido.item() else "❌ NÃO ATENDIDO"
        print(f"{nome:<18} [{lat:.5f}, {lon:.5f}] -> {status} ({conf:.1%}) {'OK' if ok else 'FALHOU'}")

    print(f"\n{acertos}/{len(casos)} casos de teste corretos.")
    return acertos == len(casos)


def plotar_fronteira(classificador: ClassificadorAreaAtendimento) -> None:
    """Salva um mapa de calor da fronteira de decisão, se matplotlib existir."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nGráfico não gerado: matplotlib não está instalado neste venv.")
        print("Para habilitar: pip install matplotlib numpy")
        return

    eixos = torch.arange(-6, 6.0001, 0.1)
    malha = torch.stack(torch.meshgrid(eixos, eixos, indexing="xy"), dim=-1).reshape(-1, 2)
    probabilidades = classificador.prever_km(malha).reshape(len(eixos), len(eixos)).numpy()

    CAMINHO_GRAFICO.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.contourf(eixos.numpy(), eixos.numpy(), probabilidades, levels=20, cmap="RdYlGn", vmin=0, vmax=1)
    plt.scatter([0], [0], color="black", marker="*", s=180, label="Loja (hub)")
    plt.xlabel("Leste (km)")
    plt.ylabel("Norte (km)")
    plt.title("Área de atendimento aprendida pela IA")
    plt.legend()
    plt.colorbar(label="Probabilidade de atendimento")
    plt.savefig(CAMINHO_GRAFICO, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGráfico salvo em: {CAMINHO_GRAFICO}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classificador de área de delivery (Aula 13.4 evoluída)")
    parser.add_argument("--plot", action="store_true", help="salva a fronteira de decisão em graficos/")
    parser.add_argument("--recarregar", action="store_true", help="usa o modelo salvo, sem treinar")
    args = parser.parse_args()

    if args.recarregar and not CAMINHO_MODELO.exists():
        print("Modelo ainda não existe. Treinando antes de recarregar...")

    if not args.recarregar or not CAMINHO_MODELO.exists():
        treinar()

    classificador = ClassificadorAreaAtendimento()
    print(f"\nModelo carregado (treinado até a época {classificador.epoca}).")
    validar_classificador(classificador)

    if args.plot:
        plotar_fronteira(classificador)


if __name__ == "__main__":
    main()
