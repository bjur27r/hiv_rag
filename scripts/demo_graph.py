# -*- coding: utf-8 -*-
"""Demo del grafo agentico (LangGraph): flujo completo + pausa/reanudacion HITL.

    python scripts/demo_graph.py            # REAL si hay claves en .env
    python scripts/demo_graph.py --simulado # fuerza modo simulado (sin gasto)
"""
import argparse
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--simulado", action="store_true")
args = ap.parse_args()
if args.simulado:  # debe ocurrir ANTES de importar (config carga .env)
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["ANTHROPIC_API_KEY"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langgraph.types import Command
from asistente_vih.agent import construir_grafo

grafo = construir_grafo()


def ejecutar(titulo, pregunta, thread, resume=None):
    print("\n" + "=" * 72); print(titulo); print("=" * 72)
    cfg = {"configurable": {"thread_id": thread}}
    entrada = {"pregunta": pregunta, "user_id": "med-demo", "thread_id": thread}
    estado = grafo.invoke(entrada, cfg)

    if "__interrupt__" in estado:  # el grafo se ha PAUSADO en HITL
        itr = estado["__interrupt__"][0].value
        print("Pregunta:", pregunta)
        print("\n[HITL — GRAFO PAUSADO]")
        print("  motivo :", itr["motivo"])
        print("  pide   :", itr["pregunta_al_medico"])
        if resume is None:
            print("\n(No se esperaba pausa aqui; el grafo queda a la espera del medico.)")
            return
        print("\n[El medico responde:", repr(resume) + "]")
        estado = grafo.invoke(Command(resume=resume), cfg)  # REANUDA desde el checkpoint

    print("Pregunta:", pregunta)
    print("Ruta    :", " -> ".join(estado.get("ruta", [])))
    v = estado.get("veredicto") or {}
    print("Veredicto fidelidad:", v.get("veredicto", "(n/a)"))
    print("\nRESPUESTA:\n" + (estado.get("respuesta", "") or "(vacia)")[:900])


ejecutar("1) Pregunta factual (flujo completo)",
         "¿A partir de que filtrado glomerular puede usarse tenofovir alafenamida?",
         thread="t-normal")

ejecutar("2) Decision seria -> pausa HITL y reanudacion",
         "¿Es seguro cambiar a cabotegravir + rilpivirina inyectable si hubo fracaso previo con ITINN y ahora esta suprimido?",
         thread="t-hitl",
         resume="Genotipo previo: sin mutaciones de resistencia a ITINN documentadas.")
