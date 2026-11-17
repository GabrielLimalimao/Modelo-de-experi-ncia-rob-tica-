"""Modelos de request/response validados pelo Pydantic."""

from pydantic import BaseModel, Field


class Coordenada(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, examples=[-23.5505])
    longitude: float = Field(..., ge=-180, le=180, examples=[-46.6333])


class AtendimentoResponse(BaseModel):
    latitude: float
    longitude: float
    atendido: bool
    probabilidade: float = Field(..., ge=0, le=1)
    raio_km: float


class LoteRequest(BaseModel):
    coordenadas: list[Coordenada] = Field(..., min_length=1, max_length=1000)


class LoteResponse(BaseModel):
    resultados: list[AtendimentoResponse]
    total: int
    atendidos: int


class InfoResponse(BaseModel):
    status: str
    epoca_treinamento: int
    perda_validacao: float | None
    hub_latitude: float
    hub_longitude: float
    raio_km: float
