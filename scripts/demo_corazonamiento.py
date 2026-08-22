"""Demo del modo 'Razonar el caso' (bucle de co-razonamiento) — SIMULADO sin claves.

Muestra el ida y vuelta: el agente busca, detecta una bifurcacion, PREGUNTA al
medico (aqui un 'medico simulado' elige una opcion), reanuda y responde citado.

  $env:PYTHONPATH = "src"
  python scripts/demo_corazonamiento.py
  python scripts/demo_corazonamiento.py "¿Que pauta de inicio recomienda la guia?"

Con --real usa las claves del .env (gasta). Por defecto corre en SIMULADO.
"""
import sys

from asistente_vih.agent import corazonamiento as cr


def medico_simulado(ev: dict) -> str:
    """Stub del medico: elige la primera opcion. En real, lo decide el medico."""
    opciones = ev.get("opciones") or ["(sin opciones)"]
    return opciones[0]


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--real"]
    real = "--real" in sys.argv
    pregunta = " ".join(args) or ("Mi paciente con insuficiencia renal avanzada, "
                                  "¿que antirretrovirales debo evitar?")
    simulado = None if real else True

    print(f"\n{'='*70}\n  Modo: Razonar el caso  ({'REAL' if real else 'SIMULADO'})\n{'='*70}")
    print(f"\n[MEDICO]  {pregunta}\n")

    ev = cr.iniciar(pregunta, thread_id="demo", simulado=simulado)
    while ev["tipo"] == "preguntar":
        print(f"[AGENTE]  {ev['razonamiento']}")
        print(f"          Criterio: {ev['criterio']}")
        for i, op in enumerate(ev["opciones"], 1):
            print(f"            [{i}] {op}")
        resp = medico_simulado(ev)
        print(f"[MEDICO]  (elige) -> {resp}\n")
        ev = cr.responder_medico("demo", resp)

    if ev["tipo"] == "respuesta":
        print(f"[AGENTE]  {ev['texto']}\n")
        print(f"          Veredicto del verificador: {ev.get('veredicto')}")
        print(f"          Fuentes citadas: {len(ev.get('citas', []))}")
    elif ev["tipo"] == "abstener":
        print(f"[AGENTE]  {ev['texto']}")
    else:
        print(f"[!]  {ev}")

    print(f"\n          Traza: {' -> '.join(ev.get('traza', []))}\n")


if __name__ == "__main__":
    main()
