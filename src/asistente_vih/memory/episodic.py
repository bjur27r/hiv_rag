"""Capa 1 - Memoria episodica: historial crudo, append-only, seudonimizado.

Cada turno de cada hilo se persiste UNA sola vez, tras pasar por el limite de
seudonimizacion. Es el sustrato de auditoria, evaluacion y mineria de huecos.
Backend de partida: un JSONL append-only (trivial de migrar a Postgres/Dynamo).

Importante: este store NO decide que se inyecta en los prompts; eso lo hace la
fachada (store.py) de forma filtrada por relevancia. Aqui solo se guarda y se lee.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path


@dataclass
class Turn:
    thread_id: str
    turn_id: int
    ts: str                     # ISO-8601
    user_id: str                # identificador seudonimo del profesional
    role: str                   # "medico" | "asistente"
    texto: str                  # YA seudonimizado

    # Metadatos del turno del asistente (vacios en turnos del medico):
    nivel: int | None = None
    guias_citadas: list[str] = field(default_factory=list)
    herramientas: list[str] = field(default_factory=list)
    ruta: str | None = None
    hitl: bool = False
    abstenido: bool = False

    # Senales de privacidad calculadas en la ingesta del turno:
    pii_directos: int = 0
    riesgo_reident: str = "bajo"

    # Feedback (se rellena despues, sobre turnos del asistente):
    feedback_pulgar: int | None = None   # +1 / -1 / None
    edicion_medico: str | None = None    # texto final tras editar el medico

    # Gobierno del dato:
    retener_hasta: str | None = None     # ISO date; fin de retencion


class EpisodicStore:
    def __init__(self, ruta: Path):
        self.ruta = ruta
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        if not self.ruta.exists():
            self.ruta.touch()

    def siguiente_turn_id(self, thread_id: str) -> int:
        return sum(1 for t in self.leer_hilo(thread_id)) + 1

    def append(self, turn: Turn) -> None:
        with open(self.ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")

    def _iter(self):
        with open(self.ruta, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield Turn(**json.loads(line))

    def leer_hilo(self, thread_id: str) -> list[Turn]:
        return sorted((t for t in self._iter() if t.thread_id == thread_id),
                      key=lambda t: t.turn_id)

    def turnos_usuario(self, user_id: str) -> list[Turn]:
        return [t for t in self._iter() if t.user_id == user_id]

    def todos(self) -> list[Turn]:
        return list(self._iter())

    def aplicar_feedback(self, thread_id: str, turn_id: int,
                         pulgar: int | None = None, edicion: str | None = None) -> bool:
        """Reescribe el JSONL aplicando feedback a un turno (operacion poco frecuente)."""
        turnos = self.todos()
        cambiado = False
        for t in turnos:
            if t.thread_id == thread_id and t.turn_id == turn_id:
                if pulgar is not None:
                    t.feedback_pulgar = pulgar
                if edicion is not None:
                    t.edicion_medico = edicion
                cambiado = True
        if cambiado:
            with open(self.ruta, "w", encoding="utf-8") as f:
                for t in turnos:
                    f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
        return cambiado

    def purgar_caducados(self, hoy: date | None = None) -> int:
        """Elimina turnos cuyo `retener_hasta` ya paso (cumplimiento de retencion)."""
        hoy = hoy or datetime.now().date()
        turnos = self.todos()
        vigentes = [t for t in turnos
                    if not t.retener_hasta or date.fromisoformat(t.retener_hasta) >= hoy]
        eliminados = len(turnos) - len(vigentes)
        if eliminados:
            with open(self.ruta, "w", encoding="utf-8") as f:
                for t in vigentes:
                    f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
        return eliminados


def fecha_retencion(dias: int, desde: datetime | None = None) -> str:
    base = (desde or datetime.now()).date()
    return (base + timedelta(days=dias)).isoformat()
