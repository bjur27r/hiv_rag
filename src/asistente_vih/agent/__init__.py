"""Orquestacion agentica con LangGraph.

Cablea las piezas ya construidas y medidas en un StateGraph con HITL y memoria:
  intake/PII -> router -> [descompone?] -> recuperar(denso) -> suficiencia
            -> sintetizar(citado) -> verificar(fidelidad) -> HITL/abstencion -> audit
"""
from .graph import construir_grafo

__all__ = ["construir_grafo"]
