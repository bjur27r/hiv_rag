import os
import json
import difflib
from datetime import datetime
from typing import TypedDict, Annotated, List, Dict, Any
import operator
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from ..knowledge.umls import UMLSClient

# Archivos de conocimiento
DICT_PATH = "artifacts/entity_dictionary.json"
TRIPLETS_PATH = "artifacts/triplets.jsonl"
CURATION_LOG_PATH = "artifacts/curation_log.jsonl"

class EstadoCuracion(TypedDict):
    entidades_pendientes: List[str]
    entidad_actual: str
    datos_entidad: dict
    propuestas_umls: List[dict]
    sinonimos_sugeridos: List[str]
    propuesta_final: dict
    decision_medica: dict  # Aprobado, Rechazado, Modificado
    ruta: Annotated[list, operator.add]

def n_scanner(state: EstadoCuracion) -> dict:
    """Busca la siguiente entidad no curada."""
    pendientes = state.get("entidades_pendientes", [])
    if not pendientes:
        try:
            with open(DICT_PATH, "r", encoding="utf-8") as f:
                dic = json.load(f)
            for k, v in dic.items():
                if not v.get("curado") and v.get("tipo_entidad") in ["Fármaco", "Condición Médica", "Mecanismo", "Farmaco", "Enfermedad", "Tratamiento"]:
                    pendientes.append(k)
        except Exception as e:
            print(f"Error cargando dic: {e}")
            
    if not pendientes:
        return {"ruta": ["fin"]}
        
    actual = pendientes.pop(0)
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        dic = json.load(f)
        
    return {
        "entidad_actual": actual, 
        "datos_entidad": dic.get(actual, {}),
        "entidades_pendientes": pendientes,
        "ruta": ["scanner"]
    }

def n_investigador(state: EstadoCuracion) -> dict:
    """Usa UMLS para buscar candidatos para la entidad actual y sugiere sinónimos."""
    actual = state["entidad_actual"]
    
    # 1. Buscar en UMLS
    cliente = UMLSClient()
    termino = actual.split("(")[0].strip()
    resultados = cliente.search_term(termino)
    top_3 = resultados[:3]
    
    propuesta = {
        "sugerencia_principal": top_3[0] if top_3 else None,
        "alternativas": top_3[1:],
        "accion": "mapear_umls" if top_3 else "requiere_busqueda_manual"
    }
    
    # 2. Buscar sinónimos en el resto de entidades no curadas usando difflib
    sinonimos = []
    pendientes = state.get("entidades_pendientes", [])
    if pendientes:
        # Buscamos coincidencias cercanas (>= 0.6 de similitud)
        matches = difflib.get_close_matches(actual, pendientes, n=10, cutoff=0.6)
        # Filtramos para no sugerir la misma entidad exacta (aunque no debería estar)
        sinonimos = [m for m in matches if m != actual]
    
    return {
        "propuestas_umls": top_3,
        "propuesta_final": propuesta,
        "sinonimos_sugeridos": sinonimos,
        "ruta": ["investigador"]
    }

def n_hitl(state: EstadoCuracion) -> dict:
    """Pausa el grafo y espera confirmación en el Front-End."""
    from langgraph.types import interrupt
    
    interrupcion = {
        "entidad": state["entidad_actual"],
        "tipo": state["datos_entidad"].get("tipo_entidad"),
        "alias": state["datos_entidad"].get("alias", []),
        "propuestas_umls": state.get("propuestas_umls", []),
        "sinonimos_sugeridos": state.get("sinonimos_sugeridos", [])
    }
    
    # PAUSA: El front-end recogerá esto y devolverá la decision_medica
    decision = interrupt(interrupcion)
    
    return {"decision_medica": decision, "ruta": ["hitl"]}

def n_updater(state: EstadoCuracion) -> dict:
    """Persiste la decisión médica y genera un log (Golden Graph)."""
    decision = state.get("decision_medica", {})
    if not decision or decision.get("accion") == "rechazar":
        return {"ruta": ["updater_rechazado"]}
        
    actual = state["entidad_actual"]
    cui = decision.get("cui")
    nombre_oficial = decision.get("nombre_oficial", actual)
    sinonimos_fusionar = decision.get("sinonimos_seleccionados", [])
    
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        dic = json.load(f)
        
    # 1. Marcar la entidad actual y cambiar nombre si aplica
    if actual in dic:
        ent_data = dic.pop(actual)
        ent_data["curado"] = True
        ent_data["codigo_umls"] = cui
        ent_data["nombre_canonico"] = nombre_oficial
        
        # Combinar alias existentes
        alias_existentes = set(ent_data.get("alias", []))
        if actual != nombre_oficial:
            alias_existentes.add(actual)
            
        # 2. Fusionar sinónimos seleccionados
        for sin in sinonimos_fusionar:
            if sin in dic:
                sin_data = dic.pop(sin)
                alias_existentes.add(sin)
                alias_existentes.update(sin_data.get("alias", []))
                
        ent_data["alias"] = list(alias_existentes)
        dic[nombre_oficial] = ent_data
        
    with open(DICT_PATH, "w", encoding="utf-8") as f:
        json.dump(dic, f, indent=2, ensure_ascii=False)
        
    # 3. Guardar en el Log de Curación (Event Sourcing Commit)
    commit = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "accion": decision.get("accion"),
        "entidad_original": actual,
        "golden_entity": {
            "nombre": nombre_oficial,
            "cui": cui
        },
        "entidades_fusionadas": sinonimos_fusionar
    }
    
    # Añadir al log JSONL
    with open(CURATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(commit, ensure_ascii=False) + "\n")
        
    # Eliminar las fusionadas de las pendientes para no volver a preguntarlas
    pendientes = state.get("entidades_pendientes", [])
    pendientes = [p for p in pendientes if p not in sinonimos_fusionar]
    
    return {"entidades_pendientes": pendientes, "ruta": ["updater_guardado"]}

def _router(state: EstadoCuracion) -> str:
    if not state.get("entidad_actual"):
        return END
    return "investigador"

def construir_refinador(checkpointer=None):
    g = StateGraph(EstadoCuracion)
    
    g.add_node("scanner", n_scanner)
    g.add_node("investigador", n_investigador)
    g.add_node("hitl", n_hitl)
    g.add_node("updater", n_updater)
    
    g.add_edge(START, "scanner")
    g.add_conditional_edges("scanner", _router, {"investigador": "investigador", END: END})
    g.add_edge("investigador", "hitl")
    g.add_edge("hitl", "updater")
    g.add_edge("updater", "scanner") # Loop!
    
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        
    return g.compile(checkpointer=checkpointer)
