"""Fachada del subsistema de memoria.

Une las dos capas y aplica las reglas del diseno:
  - Todo turno se persiste SIEMPRE a traves del limite de seudonimizacion.
  - La inyeccion de contexto esta FILTRADA POR RELEVANCIA (no se vuelca el
    historial entero en el prompt): perfil de estilo (siempre) + los turnos
    previos del MISMO hilo mas relevantes a la consulta actual.
  - Las preferencias se infieren pero se confirman antes de aplicarse.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from ..config import MEMORY_DIR, RETENCION_EPISODICA_DIAS
from . import pii
from .episodic import EpisodicStore, Turn, fecha_retencion
from .inference import Propuesta, proponer
from .profile import ProfileStore, UserProfile

_TOKEN = re.compile(r"[a-záéíóúüñ0-9]+", re.IGNORECASE)
# Palabras vacias minimas para que el solape por relevancia sea util.
_STOP = {"el", "la", "los", "las", "un", "una", "de", "del", "en", "y", "o", "a",
         "que", "con", "por", "para", "se", "su", "al", "es", "si", "como", "le",
         "lo", "mi", "tiene", "puede", "debe", "cual", "cuales", "tengo", "soy"}


def _tokens(texto: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(texto or "") if t.lower() not in _STOP and len(t) > 2}


class MemoryStore:
    def __init__(self, dir_memoria: Path | None = None,
                 retencion_dias: int = RETENCION_EPISODICA_DIAS):
        base = Path(dir_memoria) if dir_memoria else MEMORY_DIR
        base.mkdir(parents=True, exist_ok=True)
        self.episodica = EpisodicStore(base / "episodic.jsonl")
        self.perfiles = ProfileStore(base / "profiles")
        self.retencion_dias = retencion_dias

    # ---------------------------------------------------------------- Capa 1
    def registrar_turno(self, thread_id: str, user_id: str, role: str, texto: str,
                        *, nivel: int | None = None, guias_citadas: list[str] | None = None,
                        herramientas: list[str] | None = None, ruta: str | None = None,
                        hitl: bool = False, abstenido: bool = False,
                        ts: datetime | None = None) -> Turn:
        """Seudonimiza SIEMPRE y persiste el turno. Devuelve el Turn guardado."""
        res = pii.scrub(texto)
        ahora = ts or datetime.now()
        turn = Turn(
            thread_id=thread_id,
            turn_id=self.episodica.siguiente_turn_id(thread_id),
            ts=ahora.isoformat(),
            user_id=user_id,
            role=role,
            texto=res.texto,  # <- nunca el original
            nivel=nivel,
            guias_citadas=guias_citadas or [],
            herramientas=herramientas or [],
            ruta=ruta,
            hitl=hitl,
            abstenido=abstenido,
            pii_directos=res.n_directos,
            riesgo_reident=res.riesgo_reident,
            retener_hasta=fecha_retencion(self.retencion_dias, ahora),
        )
        self.episodica.append(turn)
        return turn

    def registrar_feedback(self, thread_id: str, turn_id: int,
                           pulgar: int | None = None, edicion: str | None = None) -> bool:
        edic = pii.scrub(edicion).texto if edicion else None
        return self.episodica.aplicar_feedback(thread_id, turn_id, pulgar, edic)

    # ---------------------------------------------------------------- Capa 2
    def perfil(self, user_id: str) -> UserProfile:
        return self.perfiles.cargar(user_id)

    def fijar_preferencia(self, user_id: str, clave: str, valor) -> UserProfile:
        """Preferencia EXPLICITA del usuario (la fija directamente)."""
        prof = self.perfiles.cargar(user_id)
        prof.fijar(clave, valor, fuente="explicita")
        self.perfiles.guardar(prof)
        return prof

    def proponer_actualizaciones(self, user_id: str) -> list[Propuesta]:
        """Inferir-y-confirmar: devuelve propuestas SIN aplicarlas."""
        prof = self.perfiles.cargar(user_id)
        actual = {k: p.valor for k, p in prof.preferencias.items()}
        return proponer(self.episodica.turnos_usuario(user_id), actual)

    def confirmar_propuesta(self, user_id: str, propuesta: Propuesta) -> UserProfile:
        """Aplica una propuesta tras la confirmacion del usuario."""
        prof = self.perfiles.cargar(user_id)
        prof.fijar(propuesta.clave, propuesta.valor, fuente="inferida_confirmada")
        self.perfiles.guardar(prof)
        return prof

    # ------------------------------------------------- Inyeccion por relevancia
    def turnos_relevantes(self, thread_id: str, consulta: str, k: int = 3,
                          umbral: int = 1) -> list[Turn]:
        """Top-k turnos previos del MISMO hilo con mayor solape lexico con la consulta."""
        objetivo = _tokens(consulta)
        if not objetivo:
            return []
        puntuados = []
        for t in self.episodica.leer_hilo(thread_id):
            soly = len(objetivo & _tokens(t.texto))
            if soly >= umbral:
                puntuados.append((soly, t))
        puntuados.sort(key=lambda x: (x[0], x[1].turn_id), reverse=True)
        return [t for _, t in puntuados[:k]]

    def construir_contexto_sistema(self, user_id: str, thread_id: str, consulta: str,
                                   k_turnos: int = 3, recientes_n: int = 2) -> str:
        """System context = perfil de estilo (subordinado a seguridad) + turnos relevantes.

        NO incluye el historial completo: solo lo relevante a la consulta actual.
        """
        partes: list[str] = []
        ctx_perfil = self.perfiles.cargar(user_id).como_contexto()
        if ctx_perfil:
            partes.append(ctx_perfil)

        # Recencia (ultimos turnos del hilo, captura follow-ups coreferenciales como
        # "¿y en insuficiencia renal?") + relevantes por solape lexico. Union deduplicada.
        hilo = self.episodica.leer_hilo(thread_id)
        recientes = hilo[-recientes_n:] if recientes_n else []
        relevantes = self.turnos_relevantes(thread_id, consulta, k=k_turnos)
        por_id = {t.turn_id: t for t in (recientes + relevantes)}
        turnos = sorted(por_id.values(), key=lambda t: t.turn_id)
        if turnos:
            partes.append("Contexto de esta conversacion (seudonimizado):")
            for t in turnos:
                partes.append(f"  [{t.role}] {t.texto}")
            partes.append("(Es contexto de continuidad, NO una fuente de verdad clinica: "
                          "cada afirmacion se vuelve a anclar contra las guias.)")
        return "\n".join(partes)

    # ----------------------------------------------------------------- Gobierno
    def purgar_caducados(self) -> int:
        return self.episodica.purgar_caducados()
