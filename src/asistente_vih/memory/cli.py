# -*- coding: utf-8 -*-
"""Panel de gestion de la memoria (transparencia para el profesional).

Permite VER y EDITAR el perfil, revisar las propuestas inferidas y confirmarlas,
inspeccionar el hilo episodico (ya seudonimizado), aplicar feedback, revisar los
turnos con riesgo de reidentificacion y ejecutar la purga de retencion.

La memoria es visible y corregible: no es una caja negra.

Ejemplos:
    python -m asistente_vih.memory.cli perfil show --user med-0427
    python -m asistente_vih.memory.cli perfil set  --user med-0427 --clave verbosidad --valor conciso
    python -m asistente_vih.memory.cli perfil propuestas --user med-0427
    python -m asistente_vih.memory.cli perfil confirmar  --user med-0427 --clave verbosidad
    python -m asistente_vih.memory.cli hilo show --thread consulta-aa11
    python -m asistente_vih.memory.cli privacidad --user med-0427
    python -m asistente_vih.memory.cli purgar
Use --dir para apuntar a un store concreto (por defecto: artifacts/memory).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .profile import PREFERENCIAS_PERMITIDAS
from .store import MemoryStore

_BOOL = {"true": True, "false": False, "1": True, "0": False, "si": True, "no": False}


def _coaccionar(clave: str, valor: str):
    """Convierte el valor de texto de la CLI al tipo correcto segun la lista blanca."""
    permitidos = PREFERENCIAS_PERMITIDAS.get(clave)
    if permitidos and any(isinstance(v, bool) for v in permitidos):
        if valor.lower() not in _BOOL:
            raise SystemExit(f"Valor booleano invalido: {valor!r} (usa true/false)")
        return _BOOL[valor.lower()]
    return valor


def _store(args) -> MemoryStore:
    return MemoryStore(dir_memoria=Path(args.dir) if args.dir else None)


# ------------------------------------------------------------------ perfil
def cmd_perfil_show(args):
    prof = _store(args).perfil(args.user)
    print(f"Perfil de {args.user}")
    print("-" * 60)
    if not prof.preferencias:
        print("  (sin preferencias)")
        return
    for clave, p in sorted(prof.preferencias.items()):
        print(f"  {clave:<28} = {str(p.valor):<10} [{p.fuente}]  {p.actualizado[:19]}")
    print("\nComo se inyecta en el system prompt:")
    print(prof.como_contexto())


def cmd_perfil_set(args):
    mem = _store(args)
    valor = _coaccionar(args.clave, args.valor)
    try:
        mem.fijar_preferencia(args.user, args.clave, valor)
    except ValueError as e:
        raise SystemExit(f"Rechazado: {e}")
    print(f"OK: {args.clave} = {valor!r} (explicita)")


def cmd_perfil_unset(args):
    mem = _store(args)
    prof = mem.perfil(args.user)
    prof.quitar(args.clave)
    mem.perfiles.guardar(prof)
    print(f"OK: eliminada la preferencia {args.clave!r}")


def cmd_perfil_propuestas(args):
    props = _store(args).proponer_actualizaciones(args.user)
    if not props:
        print("Sin propuestas (no hay patrones suficientes).")
        return
    print("Propuestas inferidas (NO aplicadas; confirma con 'perfil confirmar'):")
    for p in props:
        print(f"  - {p.clave} = {p.valor!r}  | {p.motivo}  | turnos {p.evidencia}")


def cmd_perfil_confirmar(args):
    mem = _store(args)
    props = mem.proponer_actualizaciones(args.user)
    elegida = next((p for p in props if p.clave == args.clave), None)
    if not elegida:
        raise SystemExit(f"No hay propuesta vigente para {args.clave!r}.")
    mem.confirmar_propuesta(args.user, elegida)
    print(f"OK confirmada: {elegida.clave} = {elegida.valor!r} (inferida_confirmada)")


# -------------------------------------------------------------------- hilo
def cmd_hilo_show(args):
    turnos = _store(args).episodica.leer_hilo(args.thread)
    if not turnos:
        print("(hilo vacio o inexistente)")
        return
    print(f"Hilo {args.thread} ({len(turnos)} turnos, seudonimizado)")
    print("-" * 60)
    for t in turnos:
        marca = []
        if t.hitl:
            marca.append("HITL")
        if t.abstenido:
            marca.append("ABST")
        if t.feedback_pulgar is not None:
            marca.append(f"pulgar={t.feedback_pulgar:+d}")
        if t.riesgo_reident != "bajo":
            marca.append(f"reident={t.riesgo_reident}")
        sfx = ("  [" + ",".join(marca) + "]") if marca else ""
        print(f"  #{t.turn_id} [{t.role}] {t.texto}{sfx}")


def cmd_hilo_feedback(args):
    ok = _store(args).registrar_feedback(args.thread, args.turn,
                                          pulgar=args.pulgar, edicion=args.edicion)
    print("OK feedback aplicado." if ok else "No se encontro el turno.")


# --------------------------------------------------------------- privacidad
def cmd_privacidad(args):
    turnos = _store(args).episodica.turnos_usuario(args.user)
    riesgo = [t for t in turnos if t.riesgo_reident in ("medio", "alto")]
    print(f"Turnos de {args.user}: {len(turnos)} | con riesgo de reidentificacion: {len(riesgo)}")
    for t in riesgo:
        print(f"  #{t.turn_id} hilo {t.thread_id} riesgo={t.riesgo_reident} "
              f"pii_directos={t.pii_directos}")
        print(f"      {t.texto}")


def cmd_purgar(args):
    n = _store(args).purgar_caducados()
    print(f"Turnos purgados por retencion: {n}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="asistente_vih.memory.cli",
                                 description="Panel de gestion de la memoria")
    ap.add_argument("--dir", help="directorio del store (def: artifacts/memory)")
    sub = ap.add_subparsers(dest="grupo", required=True)

    perfil = sub.add_parser("perfil", help="ver/editar el perfil").add_subparsers(
        dest="accion", required=True)
    p = perfil.add_parser("show"); p.add_argument("--user", required=True); p.set_defaults(fn=cmd_perfil_show)
    p = perfil.add_parser("set"); p.add_argument("--user", required=True)
    p.add_argument("--clave", required=True, choices=sorted(PREFERENCIAS_PERMITIDAS))
    p.add_argument("--valor", required=True); p.set_defaults(fn=cmd_perfil_set)
    p = perfil.add_parser("unset"); p.add_argument("--user", required=True)
    p.add_argument("--clave", required=True); p.set_defaults(fn=cmd_perfil_unset)
    p = perfil.add_parser("propuestas"); p.add_argument("--user", required=True); p.set_defaults(fn=cmd_perfil_propuestas)
    p = perfil.add_parser("confirmar"); p.add_argument("--user", required=True)
    p.add_argument("--clave", required=True); p.set_defaults(fn=cmd_perfil_confirmar)

    hilo = sub.add_parser("hilo", help="inspeccionar el historial").add_subparsers(
        dest="accion", required=True)
    p = hilo.add_parser("show"); p.add_argument("--thread", required=True); p.set_defaults(fn=cmd_hilo_show)
    p = hilo.add_parser("feedback"); p.add_argument("--thread", required=True)
    p.add_argument("--turn", type=int, required=True)
    p.add_argument("--pulgar", type=int, choices=[-1, 1])
    p.add_argument("--edicion"); p.set_defaults(fn=cmd_hilo_feedback)

    p = sub.add_parser("privacidad"); p.add_argument("--user", required=True); p.set_defaults(fn=cmd_privacidad)
    sub.add_parser("purgar").set_defaults(fn=cmd_purgar)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
