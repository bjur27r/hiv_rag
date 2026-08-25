"""Harness de evaluacion sobre 2WikiMultihopQA (subset reproduce de HippoRAG 2).

Protocolo de los papers de HippoRAG 2 / CatRAG: recuperacion ABIERTA contra el
corpus completo de 6.119 pasajes (HippoRAG/reproduce/dataset/). El campo
'context' de cada pregunta (10 parrafos con distractores) NO se usa en
recuperacion; el oro son los titulos de supporting_facts (2 pasajes por
pregunta; 4 en las bridge_comparison, donde recall@2 satura en 0,5).

Senales, todas sobre titulos de pasaje:
  - recall@k     : fraccion de titulos oro presentes en el top-k.
  - full_chain@k : TODOS los titulos oro en el top-k (Full Chain Retrieval del
                   paper de CatRAG; el analogo de recall_guia 'todas').
Se reportan globales y desglosadas por tipo de pregunta, con IC bootstrap 95%.

Splits (artifacts/wiki2/): el benchmark son las 1.000 del subset (INTOCABLES
para ajuste). El 'sondeo' (n por tipo, por defecto 50) sale del dev oficial
etiquetado (data/2wiki/train.json — pese al nombre es el dev: 12.576 con oro)
excluyendo los ids del benchmark, con su propio mini-corpus (union deduplicada
de los contextos del sondeo). El resto de ids queda como pool de calibracion.

    python -m asistente_vih.eval.wiki2 --generar-splits
    python -m asistente_vih.eval.wiki2 --bm25 benchmark
    python -m asistente_vih.eval.wiki2 --bm25 sondeo --k 2 5 10 20

Cualquier recuperador se evalua adaptandolo a `buscar(query, k) -> [titulos]`
y llamando a `evaluar()`.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict

import numpy as np

from ..config import ARTIFACTS_DIR, ROOT, DATA_DIR

SUBSET_PATH = ROOT / "HippoRAG" / "reproduce" / "dataset" / "2wikimultihopqa.json"
CORPUS_PATH = ROOT / "HippoRAG" / "reproduce" / "dataset" / "2wikimultihopqa_corpus.json"
# Pese al nombre del fichero, es el dev oficial de 2Wiki (12.576 con oro);
# los dev.json/test.json descargados son el test sin oro (inutiles).
DEV_ETIQUETADO_PATH = DATA_DIR / "2wiki" / "train.json"
WIKI2_DIR = ARTIFACTS_DIR / "wiki2"

KS_DEFECTO = (2, 5, 10, 20)
N_BOOTSTRAP = 2000
SEMILLA = 13

# Agrupacion de los 4 tipos por ESTRUCTURA de salto (ver INFORME):
ESTRUCTURA_SALTO = {
    "A sin puente":    ("comparison",),
    "B puente simple": ("compositional", "inference"),
    "C doble puente":  ("bridge_comparison",),
}


# ---------------------------------------------------------------- carga

def cargar_benchmark() -> tuple[list[dict], list[dict]]:
    preguntas = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    return preguntas, corpus


def cargar_sondeo() -> tuple[list[dict], list[dict]]:
    preguntas = json.loads((WIKI2_DIR / "sondeo_preguntas.json").read_text(encoding="utf-8"))
    corpus = json.loads((WIKI2_DIR / "sondeo_corpus.json").read_text(encoding="utf-8"))
    return preguntas, corpus


def titulos_oro(pregunta: dict) -> set[str]:
    return {t for t, _ in pregunta["supporting_facts"]}


# ---------------------------------------------------------------- splits

def generar_splits(n_por_tipo: int = 50, semilla: int = SEMILLA) -> None:
    """Sondeo estratificado por tipo desde el dev etiquetado (sin ids del
    benchmark) + su mini-corpus; el resto de ids, a calibracion."""
    WIKI2_DIR.mkdir(parents=True, exist_ok=True)
    benchmark, corpus = cargar_benchmark()
    ids_benchmark = {q["_id"] for q in benchmark}
    dev = json.loads(DEV_ETIQUETADO_PATH.read_text(encoding="utf-8"))
    candidatas = [q for q in dev if q["_id"] not in ids_benchmark and q.get("answer")]

    rng = random.Random(semilla)
    por_tipo: dict[str, list[dict]] = defaultdict(list)
    for q in sorted(candidatas, key=lambda x: x["_id"]):
        por_tipo[q["type"]].append(q)
    sondeo: list[dict] = []
    for tipo in sorted(por_tipo):
        sondeo.extend(rng.sample(por_tipo[tipo], min(n_por_tipo, len(por_tipo[tipo]))))

    # Mini-corpus: union deduplicada (por titulo) de los contextos del sondeo.
    mini: dict[str, str] = {}
    for q in sondeo:
        for titulo, frases in q["context"]:
            mini.setdefault(titulo, " ".join(frases))
    mini_corpus = [{"title": t, "text": x} for t, x in sorted(mini.items())]

    # Verificacion: el oro de cada split esta dentro de su corpus.
    fuera_bench = {t for q in benchmark for t in titulos_oro(q)} - {c["title"] for c in corpus}
    fuera_sondeo = {t for q in sondeo for t in titulos_oro(q)} - set(mini)
    if fuera_bench or fuera_sondeo:
        raise RuntimeError(f"Oro fuera de corpus: benchmark={fuera_bench} sondeo={fuera_sondeo}")

    ids_sondeo = {q["_id"] for q in sondeo}
    calibracion = [q["_id"] for q in candidatas if q["_id"] not in ids_sondeo]

    (WIKI2_DIR / "sondeo_preguntas.json").write_text(
        json.dumps(sondeo, ensure_ascii=False, indent=1), encoding="utf-8")
    (WIKI2_DIR / "sondeo_corpus.json").write_text(
        json.dumps(mini_corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    (WIKI2_DIR / "calibracion_ids.json").write_text(
        json.dumps(calibracion), encoding="utf-8")

    print(f"Benchmark: {len(benchmark)} preguntas / {len(corpus)} pasajes (no se toca)")
    print(f"Sondeo: {len(sondeo)} preguntas ({dict(Counter(q['type'] for q in sondeo))}) "
          f"/ mini-corpus {len(mini_corpus)} pasajes")
    print(f"Calibracion: {len(calibracion)} ids en {WIKI2_DIR / 'calibracion_ids.json'}")


# ---------------------------------------------------------------- metricas

def _ic_bootstrap(valores: np.ndarray, semilla: int = SEMILLA) -> tuple[float, float]:
    rng = np.random.default_rng(semilla)
    medias = rng.choice(valores, size=(N_BOOTSTRAP, len(valores)), replace=True).mean(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def evaluar(buscar, preguntas: list[dict], ks=KS_DEFECTO, nombre: str = "sistema",
            guardar: bool = True) -> dict:
    """`buscar(query, k) -> [titulos ordenados]`. Devuelve agregados y escribe
    resultados por pregunta en artifacts/wiki2/resultados_<nombre>.jsonl."""
    kmax = max(ks)
    filas = []
    for q in preguntas:
        top = list(buscar(q["question"], kmax))
        oro = titulos_oro(q)
        fila = {"_id": q["_id"], "type": q["type"], "n_oro": len(oro),
                "top": top[:kmax]}
        for k in ks:
            inter = len(oro & set(top[:k]))
            fila[f"recall@{k}"] = inter / len(oro)
            fila[f"full_chain@{k}"] = float(inter == len(oro))
        filas.append(fila)

    agg: dict = {"nombre": nombre, "n": len(filas), "por_tipo": {}}
    for k in ks:
        r = np.array([f[f"recall@{k}"] for f in filas])
        fc = np.array([f[f"full_chain@{k}"] for f in filas])
        agg[f"recall@{k}"] = float(r.mean())
        agg[f"full_chain@{k}"] = float(fc.mean())
        agg[f"recall@{k}_ic"] = _ic_bootstrap(r)
        agg[f"full_chain@{k}_ic"] = _ic_bootstrap(fc)
    for tipo in sorted({f["type"] for f in filas}):
        sel = [f for f in filas if f["type"] == tipo]
        agg["por_tipo"][tipo] = {"n": len(sel)}
        for k in ks:
            agg["por_tipo"][tipo][f"recall@{k}"] = float(
                np.mean([f[f"recall@{k}"] for f in sel]))
            agg["por_tipo"][tipo][f"full_chain@{k}"] = float(
                np.mean([f[f"full_chain@{k}"] for f in sel]))
    # Estructura de salto: A sin puente (entidades nombradas; comparison),
    # B puente simple (2 saltos; compositional+inference), C doble puente
    # (2x2 saltos, 4 oros; bridge_comparison). En 2Wiki no hay mono-salto.
    agg["por_salto"] = {}
    for etiqueta, tipos in ESTRUCTURA_SALTO.items():
        sel = [f for f in filas if f["type"] in tipos]
        if not sel:
            continue
        agg["por_salto"][etiqueta] = {"n": len(sel)}
        for k in ks:
            agg["por_salto"][etiqueta][f"recall@{k}"] = float(
                np.mean([f[f"recall@{k}"] for f in sel]))
            agg["por_salto"][etiqueta][f"full_chain@{k}"] = float(
                np.mean([f[f"full_chain@{k}"] for f in sel]))

    if guardar:
        WIKI2_DIR.mkdir(parents=True, exist_ok=True)
        ruta = WIKI2_DIR / f"resultados_{nombre}.jsonl"
        with open(ruta, "w", encoding="utf-8") as f:
            for fila in filas:
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")
        with open(WIKI2_DIR / f"agregados_{nombre}.json", "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=1)
    return agg


def imprimir(agg: dict, ks=KS_DEFECTO) -> None:
    print(f"\n== {agg['nombre']} (n={agg['n']}) ==")
    cab = "  ".join(f"     @{k}" for k in ks)
    print(f"{'':16s}{cab}")
    print("recall        " + "  ".join(f"{100*agg[f'recall@{k}']:6.1f}%" for k in ks))
    print("full_chain    " + "  ".join(f"{100*agg[f'full_chain@{k}']:6.1f}%" for k in ks))
    lo, hi = agg[f"recall@5_ic"]; flo, fhi = agg[f"full_chain@5_ic"]
    print(f"IC95 @5: recall [{100*lo:.1f}, {100*hi:.1f}]  full_chain [{100*flo:.1f}, {100*fhi:.1f}]")
    for tipo, d in agg["por_tipo"].items():
        print(f"  {tipo:18s} (n={d['n']:4d})  " + "  ".join(
            f"R@{k} {100*d[f'recall@{k}']:5.1f}% FC@{k} {100*d[f'full_chain@{k}']:5.1f}%"
            for k in (2, 5)))
    if agg.get("por_salto"):
        print("  -- por estructura de salto --")
        for et, d in agg["por_salto"].items():
            print(f"  {et:18s} (n={d['n']:4d})  R@5 {100*d['recall@5']:5.1f}%  "
                  f"FC@5 {100*d['full_chain@5']:5.1f}%")


# ---------------------------------------------------------------- BM25 humo

_TOKEN = re.compile(r"\w+")


def _tokens(texto: str) -> list[str]:
    return _TOKEN.findall(texto.lower())


class BM25Wiki:
    """BM25 autocontenido sobre un corpus [{'title','text'}] (baseline de humo:
    valida el harness y da el suelo lexico; k1/b estandar)."""

    def __init__(self, corpus: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.titulos = [c["title"] for c in corpus]
        docs = [_tokens(c["title"] + " " + c["text"]) for c in corpus]
        self.longitud = np.array([len(d) for d in docs], dtype=np.float32)
        self.media = float(self.longitud.mean())
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for term, tf in Counter(d).items():
                self.postings[term].append((i, tf))
        n = len(docs)
        self.idf = {t: math.log((n - len(p) + 0.5) / (len(p) + 0.5) + 1.0)
                    for t, p in self.postings.items()}

    def buscar(self, query: str, k: int = 10) -> list[str]:
        scores = np.zeros(len(self.titulos), dtype=np.float32)
        for term in set(_tokens(query)):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for i, tf in self.postings[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.longitud[i] / self.media)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        orden = np.argsort(-scores)[:k]
        return [self.titulos[i] for i in orden if scores[i] > 0]


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--generar-splits", action="store_true")
    ap.add_argument("--n-por-tipo", type=int, default=50)
    ap.add_argument("--bm25", choices=["benchmark", "sondeo"],
                    help="corre el baseline BM25 sobre ese split")
    ap.add_argument("--k", type=int, nargs="+", default=list(KS_DEFECTO))
    args = ap.parse_args()

    if args.generar_splits:
        generar_splits(n_por_tipo=args.n_por_tipo)
    if args.bm25:
        if args.bm25 == "benchmark":
            preguntas, corpus = cargar_benchmark()
        else:
            preguntas, corpus = cargar_sondeo()
        bm25 = BM25Wiki(corpus)
        agg = evaluar(bm25.buscar, preguntas, ks=tuple(args.k),
                      nombre=f"bm25_{args.bm25}")
        imprimir(agg, ks=tuple(args.k))


if __name__ == "__main__":
    main()
