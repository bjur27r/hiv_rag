# -*- coding: utf-8 -*-
"""Demo end-to-end del subsistema de memoria.

Ejercita: seudonimizacion al persistir, perfil explicito, inferir-y-confirmar,
contexto por relevancia y la garantia de que no se puede desactivar un guardarrail.

    python scripts/demo_memory.py
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asistente_vih.memory import MemoryStore
from asistente_vih.memory.profile import UserProfile

# Store aislado para la demo (no toca artifacts/memory de produccion).
DIR = Path(__file__).resolve().parents[1] / "artifacts" / "memory_demo"
for f in DIR.glob("**/*"):
    if f.is_file():
        f.unlink()

mem = MemoryStore(dir_memoria=DIR)
USER, THREAD = "med-0427", "consulta-aa11"
t0 = datetime(2026, 6, 11, 9, 0, 0)

print("=" * 70)
print("1) SEUDONIMIZACION AL PERSISTIR")
print("=" * 70)
turno = mem.registrar_turno(
    THREAD, USER, "medico",
    "Soy el unico paciente con VIH de mi pueblo de 800 habitantes; "
    "diagnostico en 1998, NHC 4456789, tomo abacavir. Email juan@correo.es",
    ts=t0)
print("Texto guardado :", turno.texto)
print("PII directos   :", turno.pii_directos, "| riesgo reident:", turno.riesgo_reident)
print("Retener hasta  :", turno.retener_hasta)

print("\n" + "=" * 70)
print("2) PREFERENCIA EXPLICITA + RENDER COMO CONTEXTO (subordinado a seguridad)")
print("=" * 70)
mem.fijar_preferencia(USER, "mostrar_grado_evidencia", True)
mem.fijar_preferencia(USER, "formato_cita", "compacto")
print(mem.perfil(USER).como_contexto())

print("\n" + "=" * 70)
print("3) INFERIR-Y-CONFIRMAR (3 senales de 'mas breve' -> propuesta, NO aplicada)")
print("=" * 70)
for i, txt in enumerate([
        "Hazlo mas corto por favor",
        "demasiado largo, resume",
        "mas breve la proxima",
        "y en insuficiencia renal con abacavir, que recomienda?"], start=1):
    mem.registrar_turno(THREAD, USER, "medico", txt, ts=t0 + timedelta(minutes=i))

propuestas = mem.proponer_actualizaciones(USER)
for p in propuestas:
    print(f"PROPUESTA: {p.clave} = {p.valor!r}  ({p.motivo}); evidencia turnos {p.evidencia}")
print("Perfil ANTES de confirmar:", mem.perfil(USER).valor("verbosidad"))
if propuestas:
    mem.confirmar_propuesta(USER, propuestas[0])
print("Perfil DESPUES de confirmar:", mem.perfil(USER).valor("verbosidad"),
      "(fuente:", mem.perfil(USER).preferencias["verbosidad"].fuente + ")")

print("\n" + "=" * 70)
print("4) CONTEXTO POR RELEVANCIA (no se vuelca el hilo entero)")
print("=" * 70)
ctx = mem.construir_contexto_sistema(USER, THREAD,
                                     "ajuste de abacavir en insuficiencia renal")
print(ctx)

print("\n" + "=" * 70)
print("5) SEGURIDAD POR CONSTRUCCION: no se puede desactivar un guardarrail")
print("=" * 70)
for clave, valor in [("desactivar_abstencion", True), ("verbosidad", "telegrafico")]:
    try:
        UserProfile(USER).fijar(clave, valor)
        print(f"  [FALLO] se acepto {clave}={valor!r}")
    except ValueError as e:
        print(f"  [OK] rechazado {clave}={valor!r} -> {e}")

print("\n" + "=" * 70)
print("6) RETENCION: purga de turnos caducados")
print("=" * 70)
mem.registrar_turno(THREAD, USER, "medico", "turno antiguo", ts=t0)
# Forzamos caducidad reescribiendo un retener_hasta en el pasado seria invasivo;
# en su lugar mostramos el conteo actual y que la purga respeta lo vigente.
print("Turnos en store:", len(mem.episodica.todos()),
      "| purgados hoy:", mem.purgar_caducados(), "(ninguno: retencion 365 dias)")
print("\nOK demo completada.")
