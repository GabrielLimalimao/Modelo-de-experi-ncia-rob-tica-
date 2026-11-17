"""Testes automatizados da API de atendimento."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from api.main import app  # noqa: E402

cliente = TestClient(app)


def test_health():
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json() == {"status": "ok"}


def test_raiz():
    resposta = cliente.get("/")
    assert resposta.status_code == 200
    assert resposta.json()["documentacao"] == "/docs"


def test_info():
    resposta = cliente.get("/info")
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["status"] == "ok"
    assert dados["raio_km"] == 3.0
    assert dados["epoca_treinamento"] > 0


def test_centro_e_atendido():
    resposta = cliente.post(
        "/atendimento",
        json={"latitude": -23.5505, "longitude": -46.6333},
    )
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["atendido"] is True
    assert 0 <= dados["probabilidade"] <= 1


def test_ponto_distante_nao_e_atendido():
    resposta = cliente.post(
        "/atendimento",
        json={"latitude": -23.5505, "longitude": -46.5843},
    )
    assert resposta.status_code == 200
    assert resposta.json()["atendido"] is False


@pytest.mark.parametrize("campo,valor", [("latitude", 100), ("longitude", -200)])
def test_coordenada_invalida_retorna_422(campo, valor):
    resposta = cliente.post(
        "/atendimento",
        json={"latitude": -23.5505, "longitude": -46.6333, campo: valor},
    )
    assert resposta.status_code == 422


def test_lote():
    resposta = cliente.post(
        "/atendimento/lote",
        json={
            "coordenadas": [
                {"latitude": -23.5505, "longitude": -46.6333},
                {"latitude": -23.5505, "longitude": -46.5843},
            ]
        },
    )
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["total"] == 2
    assert dados["atendidos"] == 1
    assert len(dados["resultados"]) == 2


def test_lote_vazio_retorna_422():
    resposta = cliente.post("/atendimento/lote", json={"coordenadas": []})
    assert resposta.status_code == 422
