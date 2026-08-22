"""Curacion del grafo de conocimiento: entidades, anclajes SNOMED y visor.

Todos los endpoints requieren usuario autenticado (mismo esquema que el resto
de la API). Los artefactos se resuelven con rutas absolutas via
asistente_vih.config.ARTIFACTS_DIR y los pesados (GraphML, chunks, triplets,
anclajes) se cachean en memoria con invalidacion por mtime. Las escrituras
son atomicas (tmp + os.replace) y serializadas con un lock de proceso.
"""
from __future__ import annotations

import difflib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from asistente_vih.config import ARTIFACTS_DIR
from asistente_vih.knowledge.umls import UMLSClient

from .deps import get_current_user

router = APIRouter(prefix="/curation", tags=["Curación Golden Graph"],
                   dependencies=[Depends(get_current_user)])

DICT_PATH = ARTIFACTS_DIR / "entity_dictionary.json"
CURATION_LOG_PATH = ARTIFACTS_DIR / "curation_log.jsonl"
GRAPH_PATH = ARTIFACTS_DIR / "knowledge_graph.graphml"
CHUNKS_PATH = ARTIFACTS_DIR / "chunks.jsonl"
TRIPLETS_PATH = ARTIFACTS_DIR / "triplets.jsonl"
ANCLAJES_PATH = ARTIFACTS_DIR / "anclajes.jsonl"

_LOCK = threading.Lock()   # serializa las escrituras sobre los artefactos
_CACHE: dict = {}          # ruta -> (mtime, objeto)


# ------------------------------------------------------------------ utilidades
def _cacheado(path: Path, loader):
    """Objeto cacheado para `path`; recarga solo si cambio el mtime."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    clave = str(path)
    ent = _CACHE.get(clave)
    if ent and ent[0] == mtime:
        return ent[1]
    obj = loader(path)
    _CACHE[clave] = (mtime, obj)
    return obj


def _grafo():
    import networkx as nx
    return _cacheado(GRAPH_PATH, lambda p: nx.read_graphml(str(p)))


def _chunks() -> dict:
    """chunk_id canonico (GUIA::pN::idx) -> dict del chunk, en orden de fichero."""
    def load(p):
        out = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                out[c["chunk_id"]] = c
        return out
    return _cacheado(CHUNKS_PATH, load) or {}


def _aserciones() -> dict:
    """chunk_id canonico -> lista de aserciones reificadas."""
    def load(p):
        out = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                if t.get("chunk_id"):
                    out[t["chunk_id"]] = t.get("aserciones", [])
        return out
    return _cacheado(TRIPLETS_PATH, load) or {}


def _dic() -> dict:
    return _cacheado(DICT_PATH, lambda p: json.loads(p.read_text(encoding="utf-8"))) or {}


def _anclajes() -> list:
    def load(p):
        with open(p, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    return _cacheado(ANCLAJES_PATH, load) or []


def _anclajes_por_entidad() -> dict:
    return {r.get("entidad"): r for r in _anclajes()}


def _escribir_atomico(path: Path, contenido: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(contenido)
    os.replace(tmp, path)


def _log_curacion(commit: dict):
    commit = {"timestamp": datetime.now(timezone.utc).isoformat(), **commit}
    with open(CURATION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(commit, ensure_ascii=False) + "\n")


# ------------------------------------------------- curacion de entidades (dict)
class CurateRequest(BaseModel):
    accion: str
    cui: Optional[str] = None
    nombre_oficial: Optional[str] = None
    sinonimos_seleccionados: List[str] = []


@router.get("/entities")
def get_uncurated_entities():
    """Lista de entidades del diccionario pendientes de curar."""
    try:
        pendientes = [
            {"name": k, "tipo_entidad": v.get("tipo_entidad", "Desconocido")}
            for k, v in _dic().items()
            if not v.get("curado") and v.get("tipo_entidad") in
            ["Fármaco", "Condición Médica", "Mecanismo", "Farmaco", "Enfermedad", "Tratamiento"]
        ]
        return {"entities": pendientes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity/{name}")
def get_entity_details(name: str):
    """Detalles de una entidad + sinonimos candidatos (difflib) + su anclaje SNOMED si existe."""
    dic = _dic()
    if name not in dic:
        raise HTTPException(status_code=404, detail="Entidad no encontrada")
    pendientes = [k for k, v in dic.items() if not v.get("curado") and k != name]
    matches = difflib.get_close_matches(name, pendientes, n=10, cutoff=0.6)
    return {"name": name, "datos": dic[name], "sinonimos_sugeridos": matches,
            "anclaje": _anclajes_por_entidad().get(name)}


@router.post("/entity/{name}/curate")
def curate_entity(name: str, req: CurateRequest):
    """Persiste la decision manual para la entidad (fusion de sinonimos incluida)."""
    if req.accion == "rechazar":
        return {"status": "success", "message": "Ignorada."}

    cui = req.cui
    nombre_oficial = req.nombre_oficial or name
    try:
        with _LOCK:
            dic = json.loads(DICT_PATH.read_text(encoding="utf-8"))
            if name in dic:
                ent_data = dic.pop(name)
                ent_data["curado"] = True
                ent_data["codigo_umls"] = cui
                ent_data["nombre_canonico"] = nombre_oficial
                if cui:
                    try:
                        ent_data["relaciones_umls"] = UMLSClient().get_concept_relations(cui)
                    except Exception as e:
                        print(f"Error descargando relaciones de UMLS al curar: {e}")
                alias = set(ent_data.get("alias", []))
                if name != nombre_oficial:
                    alias.add(name)
                for sin in req.sinonimos_seleccionados:
                    if sin in dic:
                        sin_data = dic.pop(sin)
                        alias.add(sin)
                        alias.update(sin_data.get("alias", []))
                ent_data["alias"] = sorted(alias)
                dic[nombre_oficial] = ent_data
            _escribir_atomico(DICT_PATH, json.dumps(dic, indent=2, ensure_ascii=False))
            _log_curacion({"tipo": "entidad", "accion": req.accion, "entidad_original": name,
                           "golden_entity": {"nombre": nombre_oficial, "cui": cui},
                           "entidades_fusionadas": req.sinonimos_seleccionados})
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/manual_search")
def manual_umls_search(term: str):
    """Busca en UMLS REST online (Top 3). Via alternativa; la local es /linker_search."""
    try:
        resultados = UMLSClient().search_term(term.split("(")[0].strip())
        return {"results": resultados[:3]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------- anclajes SNOMED (anclajes.jsonl)
# La bandeja de anclajes opera sobre las PROPUESTAS del pipeline terminologico
# (anclar_entidades + juez_anclajes). Prioridad de revision: metodo 'juez'
# (el piloto mostro fallos), luego 'fuzzy', y 'exacto' al final; dentro de cada
# metodo, primero los scores mas bajos.
_PRIORIDAD_METODO = {"juez": 0, "manual": 1, "fuzzy": 2, "exacto": 3}


@router.get("/anclajes")
def listar_anclajes(estado: str = "pendientes", limit: int = 100, offset: int = 0,
                    q: Optional[str] = None):
    """Propuestas de anclaje SNOMED/UMLS. estado: pendientes|sin_anclar|curados|rechazados."""
    regs = _anclajes()
    if not regs:
        raise HTTPException(status_code=404, detail="No existe artifacts/anclajes.jsonl (ejecuta terminologia.anclar_entidades)")

    resumen = {
        "total": len(regs),
        "pendientes": sum(1 for r in regs if not r.get("curado") and not r.get("rechazado") and not r.get("sin_anclar")),
        "sin_anclar": sum(1 for r in regs if r.get("sin_anclar") and not r.get("curado") and not r.get("rechazado")),
        "curados": sum(1 for r in regs if r.get("curado")),
        "rechazados": sum(1 for r in regs if r.get("rechazado")),
    }

    if estado == "pendientes":
        sel = [r for r in regs if not r.get("curado") and not r.get("rechazado") and not r.get("sin_anclar")]
        sel.sort(key=lambda r: (_PRIORIDAD_METODO.get(r.get("metodo"), 4),
                                float(r.get("score") or 0.0)))
    elif estado == "sin_anclar":
        sel = [r for r in regs if r.get("sin_anclar") and not r.get("curado") and not r.get("rechazado")]
    elif estado == "curados":
        sel = [r for r in regs if r.get("curado")]
    elif estado == "rechazados":
        sel = [r for r in regs if r.get("rechazado")]
    else:
        raise HTTPException(status_code=400, detail="estado invalido")

    if q:
        qn = q.lower()
        sel = [r for r in sel if qn in (r.get("entidad") or "").lower()]

    return {"resumen": resumen, "total_filtrado": len(sel),
            "anclajes": sel[offset:offset + limit]}


class AnclajeDecision(BaseModel):
    entidad: str
    accion: str                      # "aprobar" | "rechazar"
    sctid: Optional[str] = None      # anclaje manual (desde /linker_search)
    cui: Optional[str] = None
    termino: Optional[str] = None


@router.post("/anclajes/decidir")
def decidir_anclaje(req: AnclajeDecision):
    """Aprueba o rechaza una propuesta de anclaje. Con sctid/cui explicitos,
    ancla manualmente (util para los 'sin_anclar' resueltos via /linker_search).
    Los cambios se aplican al grafo al reconstruirlo (construir_grafo --solo-grafo)."""
    if req.accion not in ("aprobar", "rechazar"):
        raise HTTPException(status_code=400, detail="accion debe ser 'aprobar' o 'rechazar'")
    with _LOCK:
        try:
            with open(ANCLAJES_PATH, encoding="utf-8") as f:
                regs = [json.loads(line) for line in f if line.strip()]
        except OSError as e:
            raise HTTPException(status_code=404, detail=f"anclajes.jsonl no disponible: {e}")
        idx = next((i for i, r in enumerate(regs) if r.get("entidad") == req.entidad), None)
        if idx is None:
            raise HTTPException(status_code=404, detail="Entidad sin registro de anclaje")
        r = regs[idx]
        if req.accion == "aprobar":
            if req.sctid or req.cui:
                r.update({"sctid": req.sctid or r.get("sctid"),
                          "cui": req.cui or r.get("cui"),
                          "termino_umls": req.termino or r.get("termino_umls"),
                          "metodo": "manual", "score": 1.0})
                r.pop("sin_anclar", None)
            if not r.get("sctid") and not r.get("cui"):
                raise HTTPException(status_code=400, detail="La propuesta no tiene SCTID/CUI que aprobar")
            r["curado"] = True
            r.pop("rechazado", None)
        else:
            r["rechazado"] = True
            r["curado"] = False
        regs[idx] = r
        _escribir_atomico(ANCLAJES_PATH,
                          "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in regs))
        _log_curacion({"tipo": "anclaje", "accion": req.accion, "entidad": req.entidad,
                       "sctid": r.get("sctid"), "cui": r.get("cui"),
                       "metodo": r.get("metodo")})
    return {"status": "success", "registro": r}


@router.get("/linker_search")
def linker_search(term: str, slot: Optional[str] = None, max_n: int = 5):
    """Candidatos de la terminologia LOCAL (UMLS 2026AA SQLite, SCTID primario)."""
    try:
        from asistente_vih.terminologia.linker import get_linker
        lk = get_linker()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Terminologia local no disponible: {e}")
    try:
        cands = lk.candidatos(term, slot=slot or None, max_n=max_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"results": [{"cui": c.cui, "sctid": c.sctid, "termino": c.termino,
                         "score": round(float(c.score), 3), "sab": c.sab,
                         "metodo": c.metodo, "grupos": sorted(c.grupos or [])}
                        for c in cands]}


# ----------------------------------------------------------------- visor grafo
@router.get("/search_nodes")
def search_graph_nodes(q: str):
    """Autocompletado de nodos por id o por nombre (conceptos SNOMED)."""
    G = _grafo()
    if G is None:
        return {"results": []}
    q_lower = q.lower()
    matches = []
    for n, data in G.nodes(data=True):
        if q_lower in n.lower() or q_lower in str(data.get("nombre", "")).lower():
            matches.append(n)
            if len(matches) >= 10:
                break
    return {"results": matches}


@router.get("/graph_data")
def get_graph_data(query: Optional[str] = None, hops: int = 2):
    """Subgrafo (ego a `hops` saltos) o top-100 hubs. Nodos enriquecidos con
    diccionario, anclajes SNOMED y todos los atributos del GraphML."""
    import networkx as nx
    G = _grafo()
    if G is None:
        return {"nodes": [], "links": []}
    try:
        if query:
            ql = query.lower()
            target = next((n for n in G.nodes()
                           if ql == n.lower() or ql in n.lower()), None)
            if not target:
                return {"nodes": [], "links": []}
            sub_g = nx.ego_graph(G, target, radius=hops, undirected=True)
        else:
            degrees = dict(G.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:100]
            sub_g = G.subgraph(top_nodes)

        dic = _dic()
        anc = _anclajes_por_entidad()
        nodes = []
        for n, data in sub_g.nodes(data=True):
            n_data = dic.get(n, {})
            a = anc.get(n) or {}
            attrs = {k: v for k, v in data.items()
                     if k != "tipo" and v not in (None, "")}
            nodes.append({
                "id": n,
                "label": data.get("nombre") or n,
                "type": n_data.get("tipo_entidad", data.get("tipo", "Desconocido")),
                "curado": bool(n_data.get("curado") or a.get("curado")),
                "cui": n_data.get("codigo_umls") or a.get("cui") or "",
                "sctid": data.get("sctid") or a.get("sctid") or "",
                "anclaje_metodo": a.get("metodo", ""),
                "alias": n_data.get("alias", []),
                "attrs": attrs,
            })
        links = [{"source": u, "target": v, "label": d.get("tipo_relacion", "")}
                 for u, v, d in sub_g.edges(data=True)]
        return {"nodes": nodes, "links": links}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------- visor chunks
@router.get("/chunks")
def get_chunks_list(limit: int = 100, offset: int = 0, guia: Optional[str] = None,
                    q: Optional[str] = None):
    """Lista de fragmentos con su chunk_id CANONICO (GUIA::pN::idx)."""
    chunks = list(_chunks().values())
    if guia:
        chunks = [c for c in chunks if c.get("guia") == guia]
    if q:
        qn = q.lower()
        chunks = [c for c in chunks if qn in c.get("texto", "").lower()
                  or qn in c.get("chunk_id", "").lower()]
    total = len(chunks)
    page = chunks[offset:offset + limit]
    return {"total": total,
            "chunks": [{"id": c["chunk_id"], "guia": c.get("guia", "Desconocida"),
                        "seccion": c.get("seccion", ""), "pagina": c.get("pagina"),
                        "texto_preview": c.get("texto", "")[:100] + "..."}
                       for c in page]}


@router.get("/chunk/{chunk_id:path}")
def get_chunk_details(chunk_id: str):
    """Texto completo de un chunk (id canonico) + sus aserciones reificadas."""
    c = _chunks().get(chunk_id)
    if not c:
        raise HTTPException(status_code=404, detail="Chunk no encontrado (usa el id canonico GUIA::pN::idx)")
    return {"id": chunk_id,
            "metadata": {k: v for k, v in c.items() if k != "texto"},
            "texto": c.get("texto", ""),
            "relaciones": _aserciones().get(chunk_id, [])}
