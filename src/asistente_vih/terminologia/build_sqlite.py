"""Construye la base terminologica SQLite desde la release Full de UMLS.

Streamea MRCONSO/MRSTY/MRREL directamente desde los .nlm (zips anidados) sin
extraccion completa. Filtra a las fuentes del proyecto y a la jerarquia IS_A de
SNOMED CT (fuente unica). Registra la release para reproducibilidad.

Tablas:
  terminos (term_norm, term, cui, sctid, sab, tty, lat, ispref) + FTS trigram
  semantica(cui, tui, sty, grupo)      -- solo CUIs presentes en terminos
  jerarquia(hijo, padre)               -- IS_A de SNOMEDCT_US ya NORMALIZADA:
                                          union de CHD/isa y PAR/inverse_isa
                                          (direcciones no espejadas en MRREL),
                                          sin filas suprimidas, deduplicada
  meta     (clave, valor)              -- release, fuentes, fecha

    python -m asistente_vih.terminologia.build_sqlite \
        --meta1 data/umls_tmp/2026AA-full/2026aa-1-meta.nlm \
        --meta2 data/umls_tmp/2026AA-full/2026aa-2-meta.nlm
"""
from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
import unicodedata
import zipfile
from datetime import date

from ..config import ARTIFACTS_DIR

DB_DIR = ARTIFACTS_DIR / "terminologia"
DB_PATH = DB_DIR / "umls_2026AA.sqlite"

FUENTES = ("SCTSPA", "SNOMEDCT_US", "MSHSPA", "RXNORM")
RELEASE = "2026AA"

# Grupos semanticos (SemGroups de la NLM, subconjunto clinico; resto -> OTHER).
_GRUPOS = {
    "CHEM": ["T103", "T104", "T109", "T114", "T116", "T120", "T121", "T122",
             "T123", "T125", "T126", "T127", "T129", "T130", "T131", "T192",
             "T195", "T196", "T197", "T200"],
    "DISO": ["T019", "T020", "T033", "T037", "T046", "T047", "T048", "T049",
             "T050", "T184", "T190", "T191"],
    "PROC": ["T058", "T059", "T060", "T061", "T062", "T063", "T065"],
    "ANAT": ["T017", "T018", "T021", "T022", "T023", "T024", "T025", "T026",
             "T029", "T030", "T031"],
    "PHYS": ["T032", "T039", "T040", "T041", "T042", "T043", "T044", "T045", "T201"],
    "LIVB": ["T001", "T002", "T004", "T005", "T007", "T008", "T010", "T011",
             "T012", "T013", "T014", "T015", "T016", "T096", "T097", "T098",
             "T099", "T100", "T101", "T194", "T204"],
    "GENE": ["T028", "T085", "T086", "T087", "T088"],
    "DEVI": ["T074", "T075", "T203"],
}
TUI_A_GRUPO = {tui: g for g, tuis in _GRUPOS.items() for tui in tuis}


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    return "".join(c for c in s if not unicodedata.combining(c))


def _lineas(nlm_path: str, prefijo: str):
    """Itera lineas concatenando las partes .gz (el troceo corta a mitad de linea)."""
    z = zipfile.ZipFile(nlm_path)
    partes = sorted(n for n in z.namelist() if n.split("/")[-1].startswith(prefijo))
    resto = b""
    for parte in partes:
        print(f"  leyendo {parte} ...", flush=True)
        with z.open(parte) as raw, gzip.GzipFile(fileobj=raw) as gz:
            while True:
                bloque = gz.read(1 << 22)
                if not bloque:
                    break
                bloque = resto + bloque
                trozos = bloque.split(b"\n")
                resto = trozos.pop()
                for t in trozos:
                    yield t.decode("utf-8", errors="replace")
    if resto:
        yield resto.decode("utf-8", errors="replace")


def cargar_jerarquia(db: sqlite3.Connection, meta1: str) -> None:
    """Carga jerarquia(hijo,padre) A NIVEL SCTID desde MRHIER via AUIs.

    MRREL proyecta la jerarquia de SNOMED de forma INCOMPLETA (~267k pares
    frente a ~500k isa reales; conceptos activos como 700272008 quedan sin
    padre). MRHIER en cambio trae, por atomo y contexto, el padre inmediato
    (PAUI) de cada camino jerarquico: cobertura completa. Se resuelve
    AUI -> SCUI (SCTID) para esquivar los artefactos de la fusion CUI de UMLS.
    """
    print("MRCONSO -> mapa AUI->SCTID (SNOMEDCT_US)...")
    aui_a_sctid: dict[str, str] = {}
    for linea in _lineas(meta1, "MRCONSO.RRF"):
        c = linea.rstrip("\n").split("|")
        if len(c) >= 17 and c[11] == "SNOMEDCT_US" and c[9]:
            aui_a_sctid[c[7]] = c[9]
    print(f"  AUIs mapeados: {len(aui_a_sctid)}")

    print("MRHIER -> jerarquia SCTID (padre inmediato por contexto)...")
    db.execute("DROP TABLE IF EXISTS jerarquia")
    db.execute("CREATE TABLE jerarquia(hijo TEXT, padre TEXT)")
    pares: set[tuple[str, str]] = set()
    for linea in _lineas(meta1, "MRHIER.RRF"):
        c = linea.rstrip("\n").split("|")
        # CUI 0 AUI 1 CXN 2 PAUI 3 SAB 4 RELA 5 PTR 6 HCD 7 CVF 8
        # PAUI suele venir vacio: el padre inmediato es el ULTIMO AUI del PTR.
        if len(c) < 7 or c[4] != "SNOMEDCT_US":
            continue
        paui = c[3] or (c[6].rsplit(".", 1)[-1] if c[6] else "")
        if not paui:
            continue
        hijo, padre = aui_a_sctid.get(c[1]), aui_a_sctid.get(paui)
        if hijo and padre and hijo != padre:
            pares.add((hijo, padre))
    db.executemany("INSERT INTO jerarquia VALUES(?,?)", sorted(pares))
    db.execute("CREATE INDEX IF NOT EXISTS ix_jer_hijo ON jerarquia(hijo)")
    db.execute("CREATE INDEX IF NOT EXISTS ix_jer_padre ON jerarquia(padre)")
    db.commit()
    print(f"  pares hijo->padre unicos (SCTID): {len(pares)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta1", required=True, help=".nlm con MRCONSO/MRSTY/MRHIER")
    ap.add_argument("--meta2", help=".nlm con MRREL (ya no se usa; se acepta por compatibilidad)")
    ap.add_argument("--solo-jerarquia", action="store_true",
                    help="reconstruye solo la tabla jerarquia sobre la base existente")
    args = ap.parse_args()

    if args.solo_jerarquia:
        if not args.meta1:
            ap.error("--solo-jerarquia requiere --meta1 (MRCONSO + MRHIER)")
        db = sqlite3.connect(DB_PATH)
        cargar_jerarquia(db, args.meta1)
        db.execute("INSERT OR REPLACE INTO meta VALUES('jerarquia', "
                   "'SCTID via AUI desde MRHIER (SNOMEDCT_US)')")
        db.commit(); db.close()
        return

    DB_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE terminos(term_norm TEXT, term TEXT, cui TEXT, sctid TEXT,
                              sab TEXT, tty TEXT, lat TEXT, ispref TEXT);
        CREATE TABLE semantica(cui TEXT, tui TEXT, sty TEXT, grupo TEXT);
        CREATE TABLE meta(clave TEXT PRIMARY KEY, valor TEXT);
    """)

    # ------------------------------------------------ MRCONSO (terminos)
    print("MRCONSO -> terminos (filtro SAB/idioma/SUPPRESS)...")
    cuis_kept: set[str] = set()
    lote, n = [], 0
    for linea in _lineas(args.meta1, "MRCONSO.RRF"):
        c = linea.rstrip("\n").split("|")
        # CUI0 LAT1 TS2 LUI3 STT4 SUI5 ISPREF6 AUI7 SAUI8 SCUI9 SDUI10 SAB11 TTY12 CODE13 STR14 SRL15 SUPPRESS16
        if len(c) < 17 or c[11] not in FUENTES or c[16] != "N":
            continue
        sctid = c[9] if c[11] in ("SCTSPA", "SNOMEDCT_US") else ""
        lote.append((fold(c[14]), c[14], c[0], sctid, c[11], c[12], c[1], c[6]))
        cuis_kept.add(c[0])
        if len(lote) >= 50000:
            db.executemany("INSERT INTO terminos VALUES(?,?,?,?,?,?,?,?)", lote)
            n += len(lote); lote = []
    db.executemany("INSERT INTO terminos VALUES(?,?,?,?,?,?,?,?)", lote)
    n += len(lote)
    print(f"  terminos: {n}  |  CUIs distintos: {len(cuis_kept)}")

    # ------------------------------------------------ MRSTY (semantica)
    print("MRSTY -> semantica...")
    lote, n = [], 0
    for linea in _lineas(args.meta1, "MRSTY.RRF"):
        c = linea.rstrip("\n").split("|")
        if len(c) >= 4 and c[0] in cuis_kept:
            lote.append((c[0], c[1], c[3], TUI_A_GRUPO.get(c[1], "OTHER")))
            if len(lote) >= 50000:
                db.executemany("INSERT INTO semantica VALUES(?,?,?,?)", lote)
                n += len(lote); lote = []
    db.executemany("INSERT INTO semantica VALUES(?,?,?,?)", lote)
    n += len(lote)
    print(f"  semantica: {n}")

    cargar_jerarquia(db, args.meta1)

    # ------------------------------------------------ indices + FTS + meta
    print("Indices...")
    db.executescript("""
        CREATE INDEX ix_term_norm ON terminos(term_norm);
        CREATE INDEX ix_term_cui  ON terminos(cui);
        CREATE INDEX ix_term_sctid ON terminos(sctid);
        CREATE INDEX ix_sem_cui   ON semantica(cui);
    """)
    try:
        db.execute("CREATE VIRTUAL TABLE terminos_fts USING fts5(term_norm, content='terminos', content_rowid='rowid', tokenize='trigram')")
        db.execute("INSERT INTO terminos_fts(rowid, term_norm) SELECT rowid, term_norm FROM terminos")
        fts = "trigram"
    except sqlite3.OperationalError as e:
        print(f"  FTS trigram no disponible ({e}); solo busqueda exacta/LIKE.")
        fts = "no"

    db.executemany("INSERT INTO meta VALUES(?,?)", [
        ("release", RELEASE), ("fuentes", json.dumps(FUENTES)),
        ("fts", fts), ("fecha_construccion", date.today().isoformat()),
        ("jerarquia", "SNOMEDCT_US CHD/isa + PAR/inverse_isa normalizada, SUPPRESS=N"),
    ])
    db.commit()
    db.execute("VACUUM")
    db.close()
    print(f"\nBase construida: {DB_PATH} ({DB_PATH.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
