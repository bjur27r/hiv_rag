"""Esquema del grafo de conocimiento (GraphRAG) — Fase 4.

Define el contrato del grafo que habilita el razonamiento multi-salto (las 176
preguntas que conectan >=2 guias). En Fase 1 solo se declara el esquema; la
construccion (extraccion de tripletas desde los chunks con un LLM y validacion
humana) se implementa en Fase 4. Se deja aqui para que ingesta y grafo compartan
el mismo vocabulario de entidades desde el principio.
"""
from __future__ import annotations

from enum import Enum


class NodoTipo(str, Enum):
    FARMACO = "Farmaco"
    CLASE = "ClaseFarmacologica"     # p. ej. INI, ITIAN, IP, ITINN
    CONDICION = "Condicion"          # IR, TB, embarazo, VHB, fracaso virologico...
    POBLACION = "Poblacion"          # adultos, embarazo, pediatria, coinfeccion...
    UMBRAL = "Umbral"                # FG>=30, CD4<200, ventana PEP<72h
    RECOMENDACION = "Recomendacion"  # nodo-ancla a (guia, seccion, pagina, evidencia)
    INTERACCION = "Interaccion"
    PRUEBA = "Prueba"                # HLA-B*5701, genotipo de resistencia...


class Arista(str, Enum):
    RECOMIENDA = "RECOMIENDA"               # Recomendacion -> Farmaco/Pauta
    CONTRAINDICADO_SI = "CONTRAINDICADO_SI" # Farmaco -> Condicion/Umbral
    REQUIERE_PRUEBA = "REQUIERE_PRUEBA"     # Farmaco -> Prueba
    INTERACCIONA_CON = "INTERACCIONA_CON"   # Farmaco -> Farmaco (via Interaccion)
    AJUSTAR_SI = "AJUSTAR_SI"               # Farmaco -> Umbral (ajuste de dosis)
    PERTENECE_A = "PERTENECE_A"             # Farmaco -> Clase
    APLICA_A = "APLICA_A"                   # Recomendacion -> Poblacion
    EVIDENCIA = "EVIDENCIA"                 # Recomendacion -> nivel GRADE
    DERIVA_DE = "DERIVA_DE"                 # cualquier nodo -> Chunk (trazabilidad)


# Propiedades minimas que todo nodo Recomendacion arrastra para poder citar:
#   guia, version, fecha_vigencia, seccion, pagina, nivel_evidencia, chunk_id
# Toda arista lleva `chunk_id` de origen -> ninguna relacion del grafo existe sin
# un fragmento citable que la respalde (mismo principio de grounding que el RAG).
