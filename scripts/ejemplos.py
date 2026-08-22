# -*- coding: utf-8 -*-
"""Genera ejemplos REALES de respuesta por nivel (N1/N2/N3 directo, N4 por el grafo)."""
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asistente_vih.llm.client import LLM, MODELO_SINTESIS, MODELO_VERIFICADOR
from asistente_vih.llm.generate import generar_respuesta
from asistente_vih.llm.verify import verificar_fidelidad
from asistente_vih.retrieval.dense import DenseRetriever
from asistente_vih.retrieval.decompose import DecomposingRetriever

denso = DenseRetriever()
decomp = DecomposingRetriever(denso)
sint, verif = LLM(MODELO_SINTESIS), LLM(MODELO_VERIFICADOR)


def caso(titulo, pregunta, multi=False):
    print("\n" + "#" * 78); print(titulo); print("#" * 78)
    print("PREGUNTA:", pregunta, "\n")
    hits = (decomp if multi else denso).search(pregunta, k=20 if multi else 8)
    print("GUIAS recuperadas (top):", ", ".join(dict.fromkeys(h.guia for h in hits[:6])))
    gen = generar_respuesta(pregunta, hits, llm=sint)
    v = verificar_fidelidad(pregunta, gen["respuesta"], hits, llm=verif)
    print("\nRESPUESTA:\n" + gen["respuesta"])
    print(f"\n>> Verificador: veredicto={v.get('veredicto')} | fiel={v.get('fiel')}"
          f" | abstenida={gen.get('abstenida')}")


caso("N1 — FACTUAL DE UN SALTO",
     "¿Qué prueba genética debe realizarse obligatoriamente antes de prescribir abacavir y qué se hace si el resultado es positivo?")

caso("N2 — MULTI-RESTRICCION (poblacion + intencion)",
     "¿Qué antirretrovirales deben evitarse o ajustarse de dosis en un paciente con insuficiencia renal avanzada?")

caso("N3 — MULTI-SALTO NUMERICO (dos guias)",
     "¿Qué problemas farmacológicos surgen al tratar la tuberculosis con rifampicina en un paciente que toma inhibidores de la integrasa, y qué ajuste de dosis procede?",
     multi=True)

# --- N4 por el GRAFO (decision seria -> HITL -> reanuda) ---
print("\n" + "#" * 78); print("N4 — COMPLEJO + HITL (por el grafo)"); print("#" * 78)
from langgraph.types import Command
from asistente_vih.agent import construir_grafo
g = construir_grafo()
preg = "¿Es seguro cambiar a cabotegravir + rilpivirina inyectable si hubo fracaso previo con ITINN y ahora está suprimido?"
cfg = {"configurable": {"thread_id": "ej-n4"}}
print("PREGUNTA:", preg)
est = g.invoke({"pregunta": preg, "user_id": "ej", "thread_id": "ej-n4"}, cfg)
if "__interrupt__" in est:
    itr = est["__interrupt__"][0].value
    print("\n[HITL] pausa ->", itr["pregunta_al_medico"])
    est = g.invoke(Command(resume="Genotipo previo: sin mutaciones de resistencia a ITINN documentadas."), cfg)
print("Ruta:", " -> ".join(est.get("ruta", [])))
print("\nRESPUESTA:\n" + (est.get("respuesta", "") or ""))
print(f"\n>> Verificador: {est.get('veredicto', {}).get('veredicto')}")
