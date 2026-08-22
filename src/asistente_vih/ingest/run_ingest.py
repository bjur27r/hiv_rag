"""CLI de ingesta: data/*.pdf -> artifacts/chunks.jsonl

Uso:
    python -m asistente_vih.ingest.run_ingest
    python -m asistente_vih.ingest.run_ingest --solo TAR_ADULTOS_2022
"""
from __future__ import annotations

import argparse
from collections import Counter

from ..config import ARTIFACTS_DIR, CHUNKS_PATH, DATA_DIR, GUIAS
from .chunking import fragmentar
from .loaders import cargar_pdf


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingesta de guias GeSIDA a chunks.jsonl")
    ap.add_argument("--solo", help="codigo de una unica guia a procesar (config.GUIAS)")
    ap.add_argument("--objetivo", type=int, default=1100, help="tamanio objetivo del chunk (chars)")
    args = ap.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    guias = {args.solo: GUIAS[args.solo]} if args.solo else GUIAS

    total = 0
    por_guia: Counter[str] = Counter()
    faltan: list[str] = []

    with open(CHUNKS_PATH, "w", encoding="utf-8") as out:
        for codigo, guia in guias.items():
            pdf = DATA_DIR / guia.archivo
            if not pdf.exists():
                faltan.append(guia.archivo)
                print(f"  [!] No encontrado: {guia.archivo}")
                continue
            print(f"  -> {codigo}: {guia.archivo}")
            bloques = cargar_pdf(pdf)
            chunks = fragmentar(bloques, guia, objetivo=args.objetivo)
            for c in chunks:
                out.write(c.to_json() + "\n")
            por_guia[codigo] = len(chunks)
            total += len(chunks)

    print("\nResumen de ingesta")
    print("-" * 40)
    for codigo, n in por_guia.items():
        print(f"  {codigo:<22} {n:>5} chunks")
    print("-" * 40)
    print(f"  TOTAL {total} chunks -> {CHUNKS_PATH}")
    if faltan:
        print(f"  Guias no encontradas: {len(faltan)}")


if __name__ == "__main__":
    main()
