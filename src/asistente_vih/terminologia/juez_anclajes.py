"""Juez LLM de anclajes (Fase 2b): decide entre candidatos CON CONTEXTO.

Para cada entidad sin anclar cuyo linker devuelve candidatos (score < umbral o
ambiguos), presenta al LLM la frase original de la guia donde aparece la
entidad + los candidatos con su grupo semantico, y este elige uno o NINGUNO
(salida estructurada). Es la version con contexto del filtro semantico: el
paso 5 del pipeline de linking del plan.

Los anclajes del juez quedan marcados metodo='juez' (nivel de confianza
inferior al exacto: prioridad para la auditoria HITL del refinador).
Reescribe artifacts/anclajes.jsonl (backup .bak).

    python -m asistente_vih.terminologia.juez_anclajes [--max 100]
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from ..config import ARTIFACTS_DIR
from ..llm.client import LLM, MODELO_ROUTER
from .linker import get_linker

ANCLAJES = ARTIFACTS_DIR / "anclajes.jsonl"
TRIPLETS = ARTIFACTS_DIR / "triplets.jsonl"
CHUNKS = ARTIFACTS_DIR / "chunks.jsonl"

SYSTEM = """Eres un experto en terminologia clinica (SNOMED CT / UMLS) que resuelve entity linking \
en guias GeSIDA de VIH. Se te da una ENTIDAD extraida de las guias, el CONTEXTO donde aparece y una \
lista de CANDIDATOS de la terminologia. Elige el candidato cuyo SIGNIFICADO coincide EXACTAMENTE \
con el de la entidad en ese contexto.

Reglas estrictas:
- Elige -1 (ninguno) si ningun candidato significa lo mismo. Un candidato mas general o mas \
especifico que la entidad NO vale (p. ej. 'Tropismo Del Vih' != 'Tropismo de Especies'; \
'Infeccion por VIH-2' != 'infeccion por RSV').
- Un candidato general SI vale cuando la entidad es una variante redaccional del mismo concepto \
(p. ej. 'Carga Viral Plasmatica' -> 'carga viral' es correcto).
- Ante la duda, -1. Un anclaje erroneo es peor que ninguno."""

SCHEMA = {
    "type": "object",
    "properties": {
        "indice_elegido": {"type": "integer"},
        "confianza": {"type": "string", "enum": ["alta", "media", "baja"]},
        "motivo": {"type": "string"},
    },
    "required": ["indice_elegido", "confianza", "motivo"],
    "additionalProperties": False,
}


def _contextos_por_entidad() -> dict[str, str]:
    """Primer chunk donde cada entidad aparece en una asercion (contexto real)."""
    textos = {}
    with open(CHUNKS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                textos[c["chunk_id"]] = c.get("texto", "")
    ctx: dict[str, str] = {}
    with open(TRIPLETS, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            for a in d.get("aserciones", []):
                for campo in ("intervencion", "poblacion_diana", "resultado_esperado"):
                    ent = (a.get(campo) or "").strip().title()
                    if ent and ent not in ctx:
                        ctx[ent] = textos.get(d["chunk_id"], "")[:600]
    return ctx


def _prompt(entidad: str, slot: str | None, contexto: str, cands) -> str:
    lineas = [f"ENTIDAD: {entidad}  (tipo: {slot or 'desconocido'})",
              f"CONTEXTO de la guia:\n{contexto or '(no disponible)'}", "", "CANDIDATOS:"]
    for i, c in enumerate(cands):
        lineas.append(f"  [{i}] {c.termino}  (score={c.score}, grupos={sorted(c.grupos) or '?'}, "
                      f"fuente={c.sab}, SCTID={c.sctid or '-'})")
    lineas.append("\nElige indice_elegido (o -1 si ninguno).")
    return "\n".join(lineas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, help="limitar a N entidades (prueba)")
    ap.add_argument("--hilos", type=int, default=8)
    args = ap.parse_args()

    lk = get_linker()
    contextos = _contextos_por_entidad()
    regs = [json.loads(l) for l in open(ANCLAJES, encoding="utf-8")]

    # Fase A (secuencial, SQLite): recolectar casos juzgables con sus candidatos.
    print("Recolectando candidatos...")
    casos = []
    for i, r in enumerate(regs):
        if not r.get("sin_anclar"):
            continue
        base = r["entidad"].split("(")[0].strip()
        cands = lk.candidatos(base, slot=r["slot"], max_n=5)
        if not cands:
            continue
        for c in cands:
            c.grupos = c.grupos or lk._grupos_de(c.cui)
        casos.append((i, r, cands, contextos.get(r["entidad"], "")))
        if args.max and len(casos) >= args.max:
            break
    print(f"Casos a juzgar: {len(casos)}")

    # Fase B (paralela): el juicio LLM.
    llm = LLM(MODELO_ROUTER)

    def juzgar(caso):
        i, r, cands, ctx = caso
        try:
            v = llm.json(SYSTEM, _prompt(r["entidad"], r["slot"], ctx, cands),
                         SCHEMA, max_tokens=300,
                         simulado_valor={"indice_elegido": -1, "confianza": "baja",
                                         "motivo": "simulado"})
        except Exception as e:
            return i, None, f"error: {e}"
        idx, conf = v.get("indice_elegido", -1), v.get("confianza", "baja")
        if 0 <= idx < len(cands) and conf in ("alta", "media"):
            return i, cands[idx], v.get("motivo", "")
        return i, None, v.get("motivo", "")

    stats = Counter()
    with ThreadPoolExecutor(max_workers=args.hilos) as ex:
        for i, cand, motivo in ex.map(juzgar, casos):
            if cand is None:
                stats["rechazado"] += 1
                continue
            r = regs[i]
            r.pop("sin_anclar", None)
            r.update({"cui": cand.cui, "sctid": cand.sctid,
                      "termino_umls": cand.termino, "score": cand.score,
                      "metodo": "juez", "sab": cand.sab,
                      "motivo_juez": motivo[:200], "release": lk.release})
            stats["anclado_juez"] += 1

    shutil.copy2(ANCLAJES, ANCLAJES.with_suffix(".jsonl.bak"))
    with open(ANCLAJES, "w", encoding="utf-8") as f:
        for r in regs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_anc = sum(1 for r in regs if "cui" in r)
    print(f"\nJuez: {stats['anclado_juez']} anclados, {stats['rechazado']} rechazados")
    print(f"Cobertura total: {total_anc}/{len(regs)} ({100*total_anc/len(regs):.1f}%)")


if __name__ == "__main__":
    main()
