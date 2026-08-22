"""Linker terminologico (Fase 2): entidad textual -> candidatos SCTID/CUI.

Pipeline por entidad, en este orden estricto:
  1. busqueda exacta normalizada (term_norm)
  2. busqueda difusa FTS trigram (tolera erratas y ligaduras)
  3. candidatos SIEMPRE con score, nunca "el CUI"
  4. filtro por grupo semantico segun el slot (CHEM para farmacos, DISO para
     condiciones...) — el equivalente UMLS de la restriccion por ECL
  5. umbral: por debajo, la entidad queda SIN anclar (texto libre inofensivo)
     y es candidata a la cola HITL del refinador

La jerarquia es SNOMED puro A NIVEL SCTID (tabla jerarquia(hijo,padre) con
SCTIDs, resuelta via AUI en build_sqlite): sin artefactos de la fusion CUI de
UMLS. `verificar_jerarquia()` comprueba el par de control (Tuberculosis
pulmonar ISA ...tuberculosis...) como sanidad. Un SCTID con jerarquia se
considera ACTIVO y se prefiere al elegir el SCTID de un CUI.

Identidad primaria del anclaje: SCTID (subsuncion computable, interoperable);
CUI como atributo puente.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache

from .build_sqlite import DB_PATH, fold

# Slot del proyecto -> grupos semanticos admisibles.
GRUPOS_POR_SLOT = {
    "farmaco": {"CHEM"},
    "intervencion": {"CHEM", "PROC"},
    # LIVB incluido: en dominio infeccioso el NER etiqueta patogenos (VIH,
    # Candida, M. tuberculosis) como enfermedad, pero SNOMED los clasifica
    # como organismo.
    "condicion": {"DISO", "LIVB"},
    "poblacion": {"DISO", "LIVB", "PHYS"},
    "resultado": {"DISO", "PHYS", "PROC"},
    "organismo": {"LIVB"},
}

@dataclass
class Candidato:
    cui: str
    sctid: str
    termino: str
    score: float
    sab: str = ""
    grupos: set = field(default_factory=set)
    metodo: str = "exacto"   # exacto | fuzzy


class Linker:
    def __init__(self, db_path=DB_PATH):
        if not db_path.exists():
            raise FileNotFoundError(f"Base terminologica no construida: {db_path}")
        self.db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.fts = self.db.execute(
            "SELECT valor FROM meta WHERE clave='fts'").fetchone()[0] == "trigram"
        self.release = self.db.execute(
            "SELECT valor FROM meta WHERE clave='release'").fetchone()[0]

    # ------------------------------------------------------------ candidatos
    def _grupos_de(self, cui: str) -> set:
        return {g for (g,) in self.db.execute(
            "SELECT DISTINCT grupo FROM semantica WHERE cui=?", (cui,))}

    def _tiene_jerarquia(self, sctid: str) -> bool:
        return bool(self.db.execute(
            "SELECT 1 FROM jerarquia WHERE hijo=? OR padre=? LIMIT 1",
            (sctid, sctid)).fetchone())

    def _mejor_sctid(self, cui: str, termino_norm: str) -> tuple[str, str]:
        """SCTID preferente para un CUI.

        Prioriza: (1) SCTID con jerarquia (activo; los legados quedan
        huerfanos), (2) match del termino buscado, (3) SCTSPA.
        """
        filas = self.db.execute(
            "SELECT DISTINCT sctid, sab, term_norm FROM terminos "
            "WHERE cui=? AND sctid != '' ", (cui,)).fetchall()
        if not filas:
            return "", ""
        filas.sort(key=lambda f: (not self._tiene_jerarquia(f[0]),
                                  f[2] != termino_norm, f[1] != "SCTSPA"))
        return filas[0][0], filas[0][1]

    def candidatos(self, texto: str, slot: str | None = None,
                   max_n: int = 5) -> list[Candidato]:
        """Candidatos ordenados por score. NUNCA devuelve 'el CUI' a secas."""
        q = fold(texto.strip())
        if not q:
            return []
        vistos: dict[str, Candidato] = {}

        # 1. Exacto normalizado (score 1.0)
        for cui, term, sab in self.db.execute(
                "SELECT DISTINCT cui, term, sab FROM terminos WHERE term_norm=? LIMIT 50", (q,)):
            if cui not in vistos:
                sctid, _ = self._mejor_sctid(cui, q)
                vistos[cui] = Candidato(cui=cui, sctid=sctid, termino=term,
                                        score=1.0, sab=sab, metodo="exacto")

        # 2. Difuso FTS trigram: primero la frase entera; si no hay nada, OR de
        # palabras (cubre plurales, erratas y compuestos: "inhibidores de la
        # integrasa" -> "inhibidor de la integrasa"). El score final siempre es
        # Dice de trigramas sobre la cadena completa, con umbral 0.55.
        if len(vistos) < max_n and self.fts and len(q) >= 4:
            filas = self._fts(f'"{q}"')
            if not filas:
                palabras = [w for w in q.split() if len(w) >= 4]
                if palabras:
                    filas = self._fts(" OR ".join(f'"{w}"' for w in palabras))
            for cui, term, sab, tn in filas:
                if cui in vistos:
                    continue
                score = _dice_trigram(q, tn)
                if score >= 0.55:
                    sctid, _ = self._mejor_sctid(cui, tn)
                    vistos[cui] = Candidato(cui=cui, sctid=sctid, termino=term,
                                            score=round(score, 3), sab=sab, metodo="fuzzy")

        cands = sorted(vistos.values(), key=lambda c: -c.score)

        # 3. Filtro por grupo semantico del slot
        if slot and slot in GRUPOS_POR_SLOT:
            admisibles = GRUPOS_POR_SLOT[slot]
            filtrados = []
            for c in cands:
                c.grupos = self._grupos_de(c.cui)
                if c.grupos & admisibles:
                    filtrados.append(c)
            cands = filtrados

        return cands[:max_n]

    def _fts(self, consulta: str) -> list:
        try:
            return self.db.execute(
                "SELECT t.cui, t.term, t.sab, t.term_norm "
                "FROM terminos_fts f JOIN terminos t ON t.rowid = f.rowid "
                "WHERE terminos_fts MATCH ? ORDER BY rank LIMIT 50",
                (consulta,)).fetchall()
        except sqlite3.OperationalError:
            return []

    def anclar(self, texto: str, slot: str | None = None,
               umbral: float = 0.85) -> Candidato | None:
        """Un anclaje SOLO si hay candidato claro; si no, None (texto libre).

        Ambiguo = varios candidatos exactos de grupos distintos -> None (eso
        debe resolverlo un juez con contexto o el HITL, no un ranking ciego).
        """
        cands = self.candidatos(texto, slot=slot)
        if not cands or cands[0].score < umbral:
            return None
        exactos = [c for c in cands if c.score >= 0.999]
        if len(exactos) > 1 and not slot:
            grupos = {frozenset(self._grupos_de(c.cui)) for c in exactos}
            if len(grupos) > 1:
                return None
        return cands[0]

    # ------------------------------------------------------------- jerarquia
    def verificar_jerarquia(self) -> bool:
        """Sanidad con el caso de control del proyecto: bictegravir debe
        subsumir en <=2 saltos a un concepto de inhibidor de integrasa."""
        cand = self.anclar("bictegravir", slot="farmaco")
        if not cand or not cand.sctid:
            return False
        for sctid, _ in self.ancestros(cand.sctid, max_saltos=2):
            if "integrasa" in fold(self.nombre_sctid(sctid)):
                return True
        return False

    def ancestros(self, sctid: str, max_saltos: int = 2,
                  max_descendientes_hub: int | None = None) -> list[tuple[str, int]]:
        """(sctid_ancestro, distancia) subiendo por IS_A hasta max_saltos.

        Con max_descendientes_hub, poda los HUBS: ancestros con mas
        descendientes directos que el umbral no se devuelven (cortocircuitan
        el PPR).
        """
        frontera, out, vistos = {sctid}, [], {sctid}
        for d in range(1, max_saltos + 1):
            siguientes = set()
            for c in frontera:
                for (p,) in self.db.execute(
                        "SELECT DISTINCT padre FROM jerarquia WHERE hijo=?", (c,)):
                    if p in vistos:
                        continue
                    vistos.add(p)
                    if max_descendientes_hub is not None:
                        (nd,) = self.db.execute(
                            "SELECT COUNT(DISTINCT hijo) FROM jerarquia WHERE padre=?",
                            (p,)).fetchone()
                        if nd > max_descendientes_hub:
                            continue
                    out.append((p, d))
                    siguientes.add(p)
            frontera = siguientes
        return out

    def descendientes_directos(self, sctid: str) -> list[str]:
        return [h for (h,) in self.db.execute(
            "SELECT DISTINCT hijo FROM jerarquia WHERE padre=?", (sctid,))]

    def descendientes(self, sctid: str, max_saltos: int = 4,
                      limite: int = 2000) -> set[str]:
        """Cierre transitivo hacia abajo (para consultas de CLASE, estilo ECL:
        `<< sctid`). Acotado por saltos y por tamaño total."""
        frontera, vistos = {sctid}, set()
        for _ in range(max_saltos):
            siguientes = set()
            for s in frontera:
                for h in self.descendientes_directos(s):
                    if h not in vistos:
                        vistos.add(h)
                        siguientes.add(h)
                        if len(vistos) >= limite:
                            return vistos
            frontera = siguientes
            if not frontera:
                break
        return vistos

    def nombre_sctid(self, sctid: str, prefer_es: bool = True) -> str:
        orden = "CASE sab WHEN 'SCTSPA' THEN 0 ELSE 1 END" if prefer_es \
            else "CASE sab WHEN 'SNOMEDCT_US' THEN 0 ELSE 1 END"
        f = self.db.execute(
            f"SELECT term FROM terminos WHERE sctid=? ORDER BY {orden}, ispref DESC LIMIT 1",
            (sctid,)).fetchone()
        return f[0] if f else sctid

    def nombre(self, cui: str, prefer_es: bool = True) -> str:
        orden = "CASE sab WHEN 'SCTSPA' THEN 0 WHEN 'MSHSPA' THEN 1 ELSE 2 END" \
            if prefer_es else "CASE sab WHEN 'SNOMEDCT_US' THEN 0 ELSE 1 END"
        f = self.db.execute(
            f"SELECT term FROM terminos WHERE cui=? ORDER BY {orden}, ispref DESC LIMIT 1",
            (cui,)).fetchone()
        return f[0] if f else cui


def _dice_trigram(a: str, b: str) -> float:
    ta = {a[i:i + 3] for i in range(len(a) - 2)}
    tb = {b[i:i + 3] for i in range(len(b) - 2)}
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


@lru_cache(maxsize=1)
def get_linker() -> Linker:
    return Linker()
