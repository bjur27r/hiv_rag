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

_DS = ROOT / "HippoRAG" / "reproduce" / "dataset"
# Conjuntos de datos: subsets de reproduccion de HippoRAG 2 (= los de CatRAG),
# fuente etiquetada para el sondeo (disjunta del subset) y prefijo de los
# splits/artefactos ("" para 2Wiki por compatibilidad: 'benchmark'/'sondeo';
# 'hotpot_benchmark'/'hotpot_sondeo'; 'musique_benchmark'/'musique_sondeo').
DATASETS = {
    "2wiki": {"subset": _DS / "2wikimultihopqa.json", "corpus": _DS / "2wikimultihopqa_corpus.json",
              # Pese al nombre del fichero, es el dev oficial de 2Wiki (12.576 con oro);
              # los dev.json/test.json descargados son el test sin oro (inutiles).
              "etiquetado": DATA_DIR / "2wiki" / "train.json", "prefijo": ""},
    "hotpot": {"subset": _DS / "hotpotqa.json", "corpus": _DS / "hotpotqa_corpus.json",
               "etiquetado": DATA_DIR / "hotpotqa" / "hotpot_train_v1.1.json", "prefijo": "hotpot_"},
    "musique": {"subset": _DS / "musique.json", "corpus": _DS / "musique_corpus.json",
                "etiquetado": DATA_DIR / "musique" / "musique_ans_v1.0_train.jsonl", "prefijo": "musique_"},
}
SUBSET_PATH = DATASETS["2wiki"]["subset"]
CORPUS_PATH = DATASETS["2wiki"]["corpus"]
DEV_ETIQUETADO_PATH = DATASETS["2wiki"]["etiquetado"]
WIKI2_DIR = ARTIFACTS_DIR / "wiki2"

KS_DEFECTO = (2, 5, 10, 20)
N_BOOTSTRAP = 2000
SEMILLA = 13

# Agrupacion de los tipos por ESTRUCTURA de salto (ver INFORME), por conjunto.
ESTRUCTURA_SALTO = {
    "A sin puente":    ("comparison",),
    "B puente simple": ("compositional", "inference"),
    "C doble puente":  ("bridge_comparison",),
}
ESTRUCTURAS = {
    "2wiki": ESTRUCTURA_SALTO,
    "hotpot": {"A sin puente": ("comparison",), "B puente simple": ("bridge",)},
    "musique": {"2 saltos": ("2hop",), "3 saltos": ("3hop",), "4 saltos": ("4hop",)},
}


# ---------------------------------------------------------------- carga

def dataset_de(split: str) -> str:
    """'benchmark'/'sondeo' -> 2wiki; 'hotpot_*' -> hotpot; 'musique_*' -> musique."""
    for nombre, d in DATASETS.items():
        if d["prefijo"] and split.startswith(d["prefijo"]):
            return nombre
    return "2wiki"


def normalizar(q: dict, dataset: str) -> dict:
    """Esquema comun: _id, question, answer, type, supporting_facts [[titulo, i]].
    2Wiki y HotpotQA ya lo cumplen; MuSiQue trae id/paragraphs[is_supporting]
    y el tipo es el numero de saltos de question_decomposition."""
    if dataset == "musique":
        q = dict(q)
        q["_id"] = q.get("_id") or q["id"]
        q["type"] = f"{len(q.get('question_decomposition', []))}hop"
        q["supporting_facts"] = [[p["title"], p.get("idx", i)] for i, p in enumerate(q["paragraphs"])
                                 if p.get("is_supporting")]
    return q


def cargar_benchmark(dataset: str = "2wiki") -> tuple[list[dict], list[dict]]:
    d = DATASETS[dataset]
    preguntas = [normalizar(q, dataset) for q in json.loads(d["subset"].read_text(encoding="utf-8"))]
    corpus = json.loads(d["corpus"].read_text(encoding="utf-8"))
    return preguntas, corpus


def cargar_sondeo(dataset: str = "2wiki") -> tuple[list[dict], list[dict]]:
    pre = DATASETS[dataset]["prefijo"]
    preguntas = json.loads((WIKI2_DIR / f"{pre}sondeo_preguntas.json").read_text(encoding="utf-8"))
    corpus = json.loads((WIKI2_DIR / f"{pre}sondeo_corpus.json").read_text(encoding="utf-8"))
    return preguntas, corpus


def cargar_split(split: str) -> tuple[list[dict], list[dict]]:
    """'benchmark', 'sondeo', 'hotpot_benchmark', 'hotpot_sondeo', 'musique_*'."""
    ds = dataset_de(split)
    return cargar_sondeo(ds) if split.endswith("sondeo") else cargar_benchmark(ds)


def ruta_corpus(split: str):
    """Fichero del corpus de un split (para openie/catrag)."""
    if split.endswith("sondeo"):
        return WIKI2_DIR / f"{split}_corpus.json"
    return DATASETS[dataset_de(split)]["corpus"]


def titulos_oro(pregunta: dict) -> set[str]:
    return {t for t, _ in pregunta["supporting_facts"]}


# ---------------------------------------------------------------- splits

def _candidatas_etiquetadas(dataset: str, ids_benchmark: set[str]) -> list[dict]:
    """Preguntas etiquetadas fuera del benchmark, normalizadas. HotpotQA: solo
    nivel 'hard' (el subset de reproduccion es integramente 'hard')."""
    ruta = DATASETS[dataset]["etiquetado"]
    if dataset == "musique":
        crudas = [json.loads(l) for l in open(ruta, encoding="utf-8")]
    elif dataset == "hotpot" and not ruta.exists():
        crudas = _hotpot_desde_parquet(ruta.parent)
    else:
        crudas = json.loads(ruta.read_text(encoding="utf-8"))
    out = []
    for q in crudas:
        q = normalizar(q, dataset)
        if q["_id"] in ids_benchmark or not q.get("answer"):
            continue
        if dataset == "hotpot" and q.get("level") != "hard":
            continue
        if dataset == "musique" and not q.get("answerable", True):
            continue
        out.append(q)
    return out


def _hotpot_desde_parquet(carpeta) -> list[dict]:
    """El servidor oficial (curtis.ml.cmu.edu) no responde; el espejo de
    HuggingFace (hotpotqa/hotpot_qa, config 'distractor', split train) sirve
    parquet con columnas anidadas. Se convierte al esquema original:
    supporting_facts [[titulo, sent_id]], context [[titulo, [frases]]]."""
    import pyarrow.parquet as pq
    ficheros = sorted(carpeta.glob("train-*.parquet"))
    if not ficheros:
        raise FileNotFoundError(f"ni JSON ni parquet del train de HotpotQA en {carpeta}")
    out = []
    for f in ficheros:
        t = pq.read_table(f).to_pylist()
        for r in t:
            sf, cx = r["supporting_facts"], r["context"]
            out.append({"_id": r["id"], "question": r["question"], "answer": r["answer"],
                        "type": r["type"], "level": r["level"],
                        "supporting_facts": [[a, b] for a, b in zip(sf["title"], sf["sent_id"])],
                        "context": [[a, list(b)] for a, b in zip(cx["title"], cx["sentences"])]})
    return out


def _mini_corpus(sondeo: list[dict], dataset: str) -> dict[str, str]:
    """Union deduplicada (por titulo) de los contextos de las preguntas."""
    mini: dict[str, str] = {}
    for q in sondeo:
        if dataset == "musique":
            for p in q["paragraphs"]:
                mini.setdefault(p["title"], p["paragraph_text"])
        else:
            for titulo, frases in q["context"]:
                mini.setdefault(titulo, " ".join(frases))
    return mini


def generar_splits(n_por_tipo: int = 50, semilla: int = SEMILLA, dataset: str = "2wiki") -> None:
    """Sondeo estratificado por tipo desde el conjunto etiquetado (sin ids del
    benchmark) + su mini-corpus; el resto de ids, a calibracion. Ficheros con
    el prefijo del conjunto (2Wiki sin prefijo, por compatibilidad)."""
    WIKI2_DIR.mkdir(parents=True, exist_ok=True)
    pre = DATASETS[dataset]["prefijo"]
    benchmark, corpus = cargar_benchmark(dataset)
    ids_benchmark = {q["_id"] for q in benchmark}
    candidatas = _candidatas_etiquetadas(dataset, ids_benchmark)

    rng = random.Random(semilla)
    por_tipo: dict[str, list[dict]] = defaultdict(list)
    for q in sorted(candidatas, key=lambda x: x["_id"]):
        por_tipo[q["type"]].append(q)
    sondeo: list[dict] = []
    for tipo in sorted(por_tipo):
        sondeo.extend(rng.sample(por_tipo[tipo], min(n_por_tipo, len(por_tipo[tipo]))))

    mini = _mini_corpus(sondeo, dataset)
    mini_corpus = [{"title": t, "text": x} for t, x in sorted(mini.items())]

    # Verificacion: el oro de cada split esta dentro de su corpus.
    fuera_bench = {t for q in benchmark for t in titulos_oro(q)} - {c["title"] for c in corpus}
    fuera_sondeo = {t for q in sondeo for t in titulos_oro(q)} - set(mini)
    if fuera_bench or fuera_sondeo:
        raise RuntimeError(f"Oro fuera de corpus: benchmark={fuera_bench} sondeo={fuera_sondeo}")

    ids_sondeo = {q["_id"] for q in sondeo}
    calibracion = [q["_id"] for q in candidatas if q["_id"] not in ids_sondeo]

    (WIKI2_DIR / f"{pre}sondeo_preguntas.json").write_text(
        json.dumps(sondeo, ensure_ascii=False, indent=1), encoding="utf-8")
    (WIKI2_DIR / f"{pre}sondeo_corpus.json").write_text(
        json.dumps(mini_corpus, ensure_ascii=False, indent=1), encoding="utf-8")
    (WIKI2_DIR / f"{pre}calibracion_ids.json").write_text(
        json.dumps(calibracion), encoding="utf-8")

    print(f"{dataset} benchmark: {len(benchmark)} preguntas / {len(corpus)} pasajes (no se toca)")
    print(f"Sondeo: {len(sondeo)} preguntas ({dict(Counter(q['type'] for q in sondeo))}) "
          f"/ mini-corpus {len(mini_corpus)} pasajes")
    print(f"Calibracion: {len(calibracion)} ids en {WIKI2_DIR / f'{pre}calibracion_ids.json'}")


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
    tipos_presentes = {f["type"] for f in filas}
    estructura = next((e for e in ESTRUCTURAS.values()
                       if tipos_presentes <= {t for ts in e.values() for t in ts}), ESTRUCTURA_SALTO)
    for etiqueta, tipos in estructura.items():
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
    ap.add_argument("--dataset", choices=list(DATASETS), default="2wiki")
    ap.add_argument("--n-por-tipo", type=int, default=50)
    ap.add_argument("--bm25", help="corre el baseline BM25 sobre ese split "
                    "(benchmark, sondeo, hotpot_benchmark, hotpot_sondeo, musique_*)")
    ap.add_argument("--k", type=int, nargs="+", default=list(KS_DEFECTO))
    args = ap.parse_args()

    if args.generar_splits:
        generar_splits(n_por_tipo=args.n_por_tipo, dataset=args.dataset)
    if args.bm25:
        preguntas, corpus = cargar_split(args.bm25)
        bm25 = BM25Wiki(corpus)
        agg = evaluar(bm25.buscar, preguntas, ks=tuple(args.k),
                      nombre=f"bm25_{args.bm25}")
        imprimir(agg, ks=tuple(args.k))


if __name__ == "__main__":
    main()
