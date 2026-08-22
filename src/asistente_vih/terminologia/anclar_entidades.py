"""Anclaje masivo: entidades del grafo -> propuestas SCTID/CUI (Fase 2).

Recorre entity_dictionary.json y propone un anclaje por entidad con el Linker
(exacto -> fuzzy -> filtro por grupo semantico -> umbral). NO muta el
diccionario: escribe artifacts/anclajes.jsonl como PROPUESTAS; la aprobacion es
del medico via el refinador HITL (que ya fija codigo_umls al curar).

Prueba tambien los alias si el nombre canonico no ancla.

    python -m asistente_vih.terminologia.anclar_entidades [--umbral 0.85]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from ..config import ARTIFACTS_DIR
from .linker import get_linker

DICT_PATH = ARTIFACTS_DIR / "entity_dictionary.json"
OUT_PATH = ARTIFACTS_DIR / "anclajes.jsonl"

# tipo_entidad del NER -> slot del linker (restriccion semantica).
SLOT_POR_TIPO = {
    "Farmaco": "farmaco", "Fármaco": "farmaco",
    "Enfermedad": "condicion", "CondicionClinica": "condicion",
    "Condición Médica": "condicion",
    "Poblacion": "poblacion", "Población": "poblacion",
    "Intervencion": "intervencion", "Intervención": "intervencion",
    "Tratamiento": "intervencion",
    "ResultadoClinico": "resultado",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral", type=float, default=0.85)
    ap.add_argument("--max", type=int, help="limitar a N entidades (prueba)")
    args = ap.parse_args()

    lk = get_linker()
    with open(DICT_PATH, encoding="utf-8") as f:
        dic = json.load(f)

    stats = Counter()
    n = 0
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        for nombre, datos in dic.items():
            if args.max and n >= args.max:
                break
            n += 1
            slot = SLOT_POR_TIPO.get(datos.get("tipo_entidad", ""), None)
            variantes = [nombre] + list(datos.get("alias", []))
            cand, usado = None, nombre
            for v in variantes:
                # Quita coletillas entre parentesis ("Dolutegravir (Dtg)").
                base = v.split("(")[0].strip()
                cand = lk.anclar(base, slot=slot, umbral=args.umbral)
                if cand:
                    usado = base
                    break
            reg = {"entidad": nombre, "slot": slot, "curado": bool(datos.get("curado"))}
            if cand:
                reg.update({"cui": cand.cui, "sctid": cand.sctid,
                            "termino_umls": cand.termino, "score": cand.score,
                            "metodo": cand.metodo, "sab": cand.sab,
                            "variante_usada": usado, "release": lk.release})
                stats[f"anclado_{cand.metodo}"] += 1
                stats["con_sctid" if cand.sctid else "solo_cui"] += 1
            else:
                reg["sin_anclar"] = True
                stats["sin_anclar"] += 1
            out.write(json.dumps(reg, ensure_ascii=False) + "\n")

    print(f"Entidades procesadas: {n}  (release {lk.release})")
    for k, v in stats.most_common():
        print(f"  {k:<18} {v}  ({100*v/n:.1f}%)")
    print(f"Propuestas en {OUT_PATH}")


if __name__ == "__main__":
    main()
