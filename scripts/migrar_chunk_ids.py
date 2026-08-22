"""Migra triplets.jsonl de IDs posicionales (chunk_N) a los chunk_id canonicos.

La extraccion masiva original indexaba los chunks por posicion de linea
(chunk_0, chunk_1, ...) porque leia c.get("id") en lugar de c.get("chunk_id").
Este script reconstruye el mapeo posicion -> chunk_id canonico (estable ante
re-ingestas) y reescribe triplets.jsonl, incluidos los id_asercion derivados
(chunk_0_1 -> TAR_ADULTOS_2022::p1::0_1). Deja copia en triplets.jsonl.bak.

Uso:  python scripts/migrar_chunk_ids.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asistente_vih.config import ARTIFACTS_DIR, CHUNKS_PATH  # noqa: E402

TRIPLETS = ARTIFACTS_DIR / "triplets.jsonl"


def main():
    posmap = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                posmap[f"chunk_{i}"] = json.loads(line)["chunk_id"]
    print(f"Mapeo construido: {len(posmap)} posiciones -> chunk_id canonico")

    bak = TRIPLETS.with_suffix(".jsonl.bak")
    if not bak.exists():
        shutil.copy2(TRIPLETS, bak)
        print(f"Backup: {bak}")

    migradas, ya_ok, huerfanas = 0, 0, 0
    lineas_out = []
    with open(TRIPLETS, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            viejo = d["chunk_id"]
            if viejo in posmap:
                nuevo = posmap[viejo]
                d["chunk_id"] = nuevo
                for a in d.get("aserciones", []):
                    aid = a.get("id_asercion", "")
                    if aid.startswith(viejo + "_"):
                        a["id_asercion"] = nuevo + aid[len(viejo):]
                migradas += 1
            elif "::" in viejo:
                ya_ok += 1  # ya canonico (re-ejecucion del script)
            else:
                huerfanas += 1
                print(f"  AVISO: {viejo} sin mapeo, se conserva tal cual")
            lineas_out.append(json.dumps(d, ensure_ascii=False))

    with open(TRIPLETS, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas_out) + "\n")

    print(f"Migradas: {migradas} | ya canonicas: {ya_ok} | sin mapeo: {huerfanas}")
    print("Ahora reconstruye el grafo:  python -m asistente_vih.ingest.construir_grafo --solo-grafo")


if __name__ == "__main__":
    main()
