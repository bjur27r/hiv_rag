"""OpenIE reificado en ingles sobre el corpus 2WikiMultihopQA (Fase B).

Extrae de cada pasaje aserciones reificadas genericas (sujeto / relacion /
objeto / calificadores / frase verbalizada) + entidades, con gpt-4o-mini.
La frase ('descripcion_relacion') es lo que luego embebe embeber_aserciones:
debe ser autocontenida, con nombres completos y sin pronombres.

Cache SQLite por (modelo, version_prompt, titulo): re-lanzar no re-factura.

    python -m asistente_vih.ingest.wiki2_openie --split sondeo --muestra 20
    python -m asistente_vih.ingest.wiki2_openie --split sondeo
    python -m asistente_vih.ingest.wiki2_openie --split benchmark
    python -m asistente_vih.ingest.wiki2_openie --cobertura   # diagnostico vs triples oro

Salida: artifacts/wiki2/openie_<split>.jsonl con
  {"chunk_id": titulo, "entidades": [...], "aserciones": [
      {"id_asercion", "sujeto", "relacion_base", "objeto",
       "calificadores", "descripcion_relacion"}]}
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from ..config import ARTIFACTS_DIR

WIKI2_DIR = ARTIFACTS_DIR / "wiki2"
CACHE_PATH = WIKI2_DIR / "openie_cache.sqlite"
MODELO = "gpt-4o-mini"
VERSION_PROMPT = "v1"
HILOS = 8

_PROMPT = """You extract facts from encyclopedia passages to build a knowledge graph.

Return STRICT JSON only: {"entities": [...], "assertions": [...]}

- "entities": every named entity mentioned in the passage (people, creative works, places, organizations). Use the fullest name form the passage gives.
- "assertions": one item per atomic fact stated in the passage:
  {"subject": "<entity>",
   "relation": "<short UPPER_SNAKE relation, e.g. DIRECTED, MOTHER_OF, FATHER_OF, SPOUSE_OF, DATE_OF_BIRTH, DATE_OF_DEATH, PLACE_OF_BIRTH, NATIONALITY, OCCUPATION, RELEASE_DATE, BASED_ON, FOUNDED, MEMBER_OF>",
   "object": "<entity or literal value (date, place, occupation)>",
   "qualifiers": {"date": "...", "place": "..."},
   "sentence": "<ONE self-contained sentence stating the fact, full names, no pronouns>"}

Rules:
- Extract ALL facts: family relations, birth/death dates and places, creator/work links, release/publication dates, nationality, occupation, positions held.
- Dates and places attached to a fact go BOTH in the object/qualifiers and spelled out in the sentence.
- Omit "qualifiers" keys that are empty. No commentary, JSON only."""


# ---------------------------------------------------------------- cache

def _abrir_cache() -> sqlite3.Connection:
    WIKI2_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE_PATH, check_same_thread=False)
    con.execute("CREATE TABLE IF NOT EXISTS openie ("
                "clave TEXT PRIMARY KEY, respuesta TEXT)")
    con.commit()
    return con


def _clave(titulo: str) -> str:
    return f"{MODELO}|{VERSION_PROMPT}|{titulo}"


# ---------------------------------------------------------------- extraccion

def _extraer_uno(client: OpenAI, con: sqlite3.Connection, lock: threading.Lock,
                 pasaje: dict) -> dict | None:
    titulo, texto = pasaje["title"], pasaje["text"]
    with lock:
        fila = con.execute("SELECT respuesta FROM openie WHERE clave=?",
                           (_clave(titulo),)).fetchone()
    if fila:
        crudo = fila[0]
    else:
        r = client.chat.completions.create(
            model=MODELO, temperature=0.0,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": _PROMPT},
                      {"role": "user", "content": f"Passage title: {titulo}\n\n{texto}"}])
        crudo = r.choices[0].message.content
        with lock:
            con.execute("INSERT OR REPLACE INTO openie VALUES (?,?)",
                        (_clave(titulo), crudo))
            con.commit()
    try:
        datos = json.loads(crudo)
    except json.JSONDecodeError:
        return {"chunk_id": titulo, "entidades": [], "aserciones": [],
                "error": "json_invalido"}
    aserciones = []
    for i, a in enumerate(datos.get("assertions", [])):
        if not isinstance(a, dict):
            continue
        aserciones.append({
            "id_asercion": f"{titulo}::a{i}",
            "sujeto": str(a.get("subject", "")),
            "relacion_base": str(a.get("relation", "")),
            "objeto": str(a.get("object", "")),
            "calificadores": a.get("qualifiers") or {},
            "descripcion_relacion": str(a.get("sentence", "")),
        })
    entidades = [str(e) for e in datos.get("entities", []) if e]
    return {"chunk_id": titulo, "entidades": entidades, "aserciones": aserciones}


def extraer(split: str, muestra: int | None = None) -> None:
    if split == "sondeo":
        corpus = json.loads((WIKI2_DIR / "sondeo_corpus.json").read_text(encoding="utf-8"))
    else:
        from ..eval.wiki2 import CORPUS_PATH
        corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    if muestra:
        corpus = corpus[:muestra]
    client, con, lock = OpenAI(), _abrir_cache(), threading.Lock()
    salida = WIKI2_DIR / f"openie_{split}.jsonl"
    resultados: dict[str, dict] = {}
    errores = 0
    with ThreadPoolExecutor(max_workers=HILOS) as pool:
        futuros = {pool.submit(_extraer_uno, client, con, lock, p): p["title"]
                   for p in corpus}
        for n, fut in enumerate(as_completed(futuros), 1):
            res = fut.result()
            if res:
                resultados[res["chunk_id"]] = res
                errores += 1 if res.get("error") else 0
            if n % 200 == 0 or n == len(corpus):
                print(f"  {n}/{len(corpus)} pasajes (errores json: {errores})")
    # orden estable = orden del corpus
    with open(salida, "w", encoding="utf-8") as f:
        for p in corpus:
            if p["title"] in resultados:
                f.write(json.dumps(resultados[p["title"]], ensure_ascii=False) + "\n")
    n_aser = sum(len(r["aserciones"]) for r in resultados.values())
    n_ent = sum(len(r["entidades"]) for r in resultados.values())
    print(f"OK {salida}: {len(resultados)} pasajes, {n_aser} aserciones, "
          f"{n_ent} menciones de entidad, {errores} errores json")


# ---------------------------------------------------------------- cobertura

def _norm(s: str) -> str:
    import re
    s = re.sub(r"\(.*?\)", " ", s.lower())
    return re.sub(r"[^0-9a-zñáéíóúü]+", " ", s).strip()


def _coincide(entidad_norm: str, tokens_texto: set[str]) -> bool:
    """Matching por solape de tokens (>=60%): tolera formas de nombre distintas
    ('Edward III of England' vs 'Edward III') y ordenes de fecha
    ('8 February 1964' vs 'February 8, 1964'), que como substring fallan."""
    toks = entidad_norm.split()
    if not toks:
        return False
    presentes = sum(1 for t in toks if t in tokens_texto)
    return presentes / len(toks) >= 0.6


def cobertura_sondeo() -> None:
    """% de triples oro (evidences) capturados por las aserciones extraidas de
    los pasajes oro de cada pregunta del sondeo. Estricta: sujeto y objeto del
    triple en la MISMA asercion; laxa: cada uno en alguna asercion/entidad.
    Es el techo de recuperacion del KG: lo que no se extrajo no se recupera."""
    preguntas = json.loads((WIKI2_DIR / "sondeo_preguntas.json").read_text(encoding="utf-8"))
    extraccion = {}
    with open(WIKI2_DIR / "openie_sondeo.jsonl", encoding="utf-8") as f:
        for linea in f:
            d = json.loads(linea)
            extraccion[d["chunk_id"]] = d
    tot = estricta = laxa = 0
    preg_ok = 0
    sin_extraccion = set()
    por_tipo: dict[str, list[int]] = {}
    for q in preguntas:
        oros = {t for t, _ in q["supporting_facts"]}
        docs = [extraccion[t] for t in oros if t in extraccion]
        sin_extraccion |= {t for t in oros if t not in extraccion}
        tokens_aser = [set(_norm(" ".join([a["sujeto"], a["objeto"],
                                           a["descripcion_relacion"],
                                           json.dumps(a["calificadores"])])).split())
                       for d in docs for a in d["aserciones"]]
        tokens_todo = set().union(*tokens_aser) if tokens_aser else set()
        tokens_todo |= {t for d in docs for e in d["entidades"]
                        for t in _norm(e).split()}
        q_ok = True
        for suj, _rel, obj in q.get("evidences", []):
            tot += 1
            ns, no = _norm(suj), _norm(obj)
            e = any(_coincide(ns, t) and _coincide(no, t) for t in tokens_aser)
            l = _coincide(ns, tokens_todo) and _coincide(no, tokens_todo)
            estricta += e
            laxa += l
            q_ok &= e
        preg_ok += q_ok
        por_tipo.setdefault(q["type"], []).append(q_ok)
    print(f"Triples oro: {tot} | estricta (misma asercion): {100*estricta/tot:.1f}% "
          f"| laxa (en el conjunto): {100*laxa/tot:.1f}%")
    print(f"Preguntas con TODOS sus triples (estricta): {100*preg_ok/len(preguntas):.1f}%")
    for tipo, vals in sorted(por_tipo.items()):
        print(f"  {tipo:18s} preguntas completas: {100*sum(vals)/len(vals):5.1f}%")
    if sin_extraccion:
        print(f"AVISO: {len(sin_extraccion)} pasajes oro sin extraccion")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", choices=["sondeo", "benchmark"])
    ap.add_argument("--muestra", type=int, help="solo los N primeros pasajes")
    ap.add_argument("--cobertura", action="store_true",
                    help="diagnostico de cobertura vs triples oro del sondeo")
    args = ap.parse_args()
    if args.split:
        extraer(args.split, muestra=args.muestra)
    if args.cobertura:
        cobertura_sondeo()


if __name__ == "__main__":
    main()
