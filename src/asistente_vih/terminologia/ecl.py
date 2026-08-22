"""Agente ECL de subsuncion (Sprint 3): consultas de CLASE con respuesta EXHAUSTIVA.

Para preguntas como "¿que INSTI estan contraindicados en embarazo?" el top-k de
cualquier retriever es la herramienta equivocada: la respuesta correcta es un
CONJUNTO. Este agente:

  1. traduce la consulta a una restriccion estilo ECL (clase + filtros) con un
     LLM barato de salida estructurada;
  2. resuelve `<< clase` (descendientes transitivos) sobre la jerarquia SNOMED
     local (SQLite, deterministico);
  3. interseca con las entidades ANCLADAS del corpus (anclajes.jsonl);
  4. recopila las recomendaciones etiquetadas que mencionan cada miembro,
     con filtro opcional de polaridad y de poblacion;
  5. devuelve la tabla exhaustiva, INCLUYENDO los miembros de la clase sin
     recomendacion en el corpus (la exhaustividad negativa tambien es respuesta).

La exhaustividad es verificable sin juicio humano: el denominador lo da la
terminologia.
"""
from __future__ import annotations

import json
from collections import defaultdict

from ..config import ARTIFACTS_DIR
from ..llm.client import LLM, MODELO_ROUTER
from .linker import get_linker

ANCLAJES = ARTIFACTS_DIR / "anclajes.jsonl"
RECS = ARTIFACTS_DIR / "recomendaciones.jsonl"

SYS = """Analiza si la consulta clinica es una PREGUNTA DE CLASE: pide informacion sobre TODOS
los miembros de una clase farmacologica o de conceptos (p. ej. "¿que INSTI...?",
"¿que farmacos antirretrovirales...?", "¿que ITIAN requieren...?").

Si lo es, devuelve:
- clase: el nombre de la clase en español, como aparece en terminologia clinica
  (p. ej. "inhibidor de la integrasa", "inhibidores de la transcriptasa inversa analogos de nucleosidos")
- poblacion: la poblacion o condicion que restringe, si la hay (p. ej. "embarazo", "insuficiencia renal"), o ""
- polaridad: "no_recomendado" si pregunta por contraindicaciones/evitar; "recomendado" si pregunta
  por recomendados/de eleccion; "" si pide todo.
Si NO es pregunta de clase, aplica=false."""

SCHEMA = {
    "type": "object",
    "properties": {
        "aplica": {"type": "boolean"},
        "clase": {"type": "string"},
        "poblacion": {"type": "string"},
        "polaridad": {"type": "string", "enum": ["", "recomendado", "no_recomendado"]},
    },
    "required": ["aplica", "clase", "poblacion", "polaridad"],
    "additionalProperties": False,
}

_NEG = ("no_recomendado", "generalmente_no_recomendado")


def _cargar_anclajes() -> dict[str, list[str]]:
    """sctid -> [nombres de entidad del corpus]"""
    por_sctid = defaultdict(list)
    with open(ANCLAJES, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("sctid"):
                    por_sctid[r["sctid"]].append(r["entidad"])
    return por_sctid


def _cargar_recs() -> list[dict]:
    return [json.loads(l) for l in open(RECS, encoding="utf-8") if l.strip()]


def consultar(pregunta: str, llm: LLM | None = None) -> dict:
    """Devuelve {aplica, clase, miembros: [...], sin_recomendacion: [...]} o {aplica: False}."""
    llm = llm or LLM(MODELO_ROUTER)
    plan = llm.json(SYS, pregunta, SCHEMA, max_tokens=200,
                    simulado_valor={"aplica": False, "clase": "", "poblacion": "", "polaridad": ""})
    if not plan.get("aplica") or not plan.get("clase"):
        return {"aplica": False}

    from ..retrieval.analyzer import analizar, fold

    lk = get_linker()
    # Tokens DISTINTIVOS de la clase: evitan que el fuzzy case otra clase
    # ("inhibidor de la MAO" no contiene "integrasa").
    toks_clase = set(analizar(plan["clase"]))

    def _es_de_clase(nombre_concepto: str) -> bool:
        return toks_clase <= set(analizar(nombre_concepto))

    # Via 1 (por nombre): conceptos candidatos al nombre de la clase cuyo
    # termino contiene TODOS los tokens distintivos; union de sus cierres.
    cands = [c for c in lk.candidatos(plan["clase"], slot="farmaco", max_n=8)
             if c.sctid and _es_de_clase(c.termino)] \
        or [c for c in lk.candidatos(plan["clase"], max_n=8)
            if c.sctid and _es_de_clase(c.termino)]
    sctids_clase = set()
    for c in cands:
        sctids_clase |= lk.descendientes(c.sctid, max_saltos=4) | {c.sctid}
    cand = cands[0] if cands else None

    # Via 2 (por subsuncion, la robusta): una entidad anclada del corpus es
    # miembro si ALGUN ancestro (<=3 saltos) lleva los tokens de la clase.
    # SNOMED reparte las clases en varios ejes (producto / sustancia-por-
    # disposicion); esta via los cubre todos sin depender del anclaje del nombre.
    por_sctid = _cargar_anclajes()
    miembros = {}   # nombre_entidad -> sctid
    for sctid, ents in por_sctid.items():
        en_clase = sctid in sctids_clase
        if not en_clase:
            for anc, _ in lk.ancestros(sctid, max_saltos=3, max_descendientes_hub=500):
                if _es_de_clase(lk.nombre_sctid(anc)):
                    en_clase = True
                    break
        if en_clase:
            for ent in ents:
                # Excluir las entidades que SON el nombre de la clase (ruido).
                if not _es_de_clase(ent) or len(set(analizar(ent)) - toks_clase) > 0:
                    miembros[ent] = sctid

    recs = _cargar_recs()
    pobl = (plan.get("poblacion") or "").lower()
    filtro_pol = plan.get("polaridad") or ""
    resultado, con_rec = [], set()
    for ent, sctid in sorted(miembros.items()):
        ent_l = ent.lower()
        encontradas = []
        for r in recs:
            txt = r["texto"].lower()
            if ent_l not in txt or len(ent_l) < 4:
                continue
            if pobl and pobl not in txt:
                continue
            if filtro_pol == "no_recomendado" and r["accion_deontica"] not in _NEG:
                continue
            if filtro_pol == "recomendado" and r["accion_deontica"] != "recomendado":
                continue
            encontradas.append({"guia": r["guia"], "pagina": r["pagina"],
                                "accion": r["accion_deontica"], "grado": r.get("grado"),
                                "vigencia": r.get("fecha_vigencia"),
                                "texto": r["texto"][:280]})
        if encontradas:
            con_rec.add(ent)
            resultado.append({"entidad": ent, "sctid": sctid,
                              "recomendaciones": encontradas[:5]})

    return {"aplica": True, "clase": plan["clase"],
            "sctid_clase": cand.sctid if cand else "",
            "poblacion": plan.get("poblacion"), "polaridad": filtro_pol,
            "n_clase_en_terminologia": len(sctids_clase),
            "n_miembros_en_corpus": len(miembros),
            "miembros": resultado,
            "sin_recomendacion_que_cumpla": sorted(set(miembros) - con_rec)}


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "¿Qué inhibidores de la integrasa se recomiendan en embarazo?"
    out = consultar(q)
    print(json.dumps(out, ensure_ascii=False, indent=2)[:4000])
