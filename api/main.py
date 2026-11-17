"""API REST para o classificador de área de atendimento (estilo iFood).

O modelo é o mesmo treinado em ``4_caixeiro_viajante.py``. Aqui ele só é
carregado e exposto via HTTP, sem nenhuma regra de negócio duplicada.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi import FastAPI

from .schemas import AtendimentoResponse, Coordenada, InfoResponse, LoteRequest, LoteResponse

RAIZ = Path(__file__).resolve().parents[1]
CAMINHO_SCRIPT_MODELO = RAIZ / "4_caixeiro_viajante.py"


def carregar_script_modelo():
    """Importa o script da aula pelo caminho (o nome começa com número)."""
    especificacao = importlib.util.spec_from_file_location(
        "caixeiro_viajante", CAMINHO_SCRIPT_MODELO
    )
    if especificacao is None or especificacao.loader is None:
        raise RuntimeError(f"Não foi possível carregar {CAMINHO_SCRIPT_MODELO}")

    modulo = importlib.util.module_from_spec(especificacao)
    sys.modules[especificacao.name] = modulo
    especificacao.loader.exec_module(modulo)
    return modulo


caixeiro = carregar_script_modelo()

if not caixeiro.CAMINHO_MODELO.exists():
    print("Modelo não encontrado. Treinando antes de subir a API...")
    caixeiro.treinar()

classificador = caixeiro.ClassificadorAreaAtendimento()

app = FastAPI(
    title="API de Área de Atendimento",
    description="Classifica se uma coordenada está dentro da região de delivery.",
    version="1.0.0",
)


def classificar(coordenada: Coordenada) -> AtendimentoResponse:
    """Converte a saída do modelo em resposta serializável."""
    probabilidade, atendido = classificador.prever(
        coordenada.latitude, coordenada.longitude
    )
    return AtendimentoResponse(
        latitude=coordenada.latitude,
        longitude=coordenada.longitude,
        atendido=bool(atendido.item()),
        probabilidade=round(probabilidade.item(), 4),
        raio_km=classificador.raio_km,
    )


@app.get("/", tags=["saúde"])
def raiz():
    return {"mensagem": "API de área de atendimento", "documentacao": "/docs"}


@app.get("/health", tags=["saúde"])
def health():
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse, tags=["saúde"])
def info():
    return InfoResponse(
        status="ok",
        epoca_treinamento=classificador.epoca,
        perda_validacao=classificador.perda_validacao,
        hub_latitude=classificador.hub_lat,
        hub_longitude=classificador.hub_lon,
        raio_km=classificador.raio_km,
    )


@app.post("/atendimento", response_model=AtendimentoResponse, tags=["classificação"])
def atendimento(coordenada: Coordenada):
    return classificar(coordenada)


@app.post("/atendimento/lote", response_model=LoteResponse, tags=["classificação"])
def atendimento_lote(lote: LoteRequest):
    resultados = [classificar(coordenada) for coordenada in lote.coordenadas]
    return LoteResponse(
        resultados=resultados,
        total=len(resultados),
        atendidos=sum(resultado.atendido for resultado in resultados),
    )
