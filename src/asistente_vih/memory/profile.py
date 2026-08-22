"""Capa 2 - Perfil destilado del usuario: preferencias de FORMA, editables.

Seguridad POR CONSTRUCCION: solo existen las claves de la lista blanca
PREFERENCIAS_PERMITIDAS, y todas son de presentacion. No hay ninguna clave que
permita desactivar un guardarrail (cita, abstencion, HITL) ni alterar el
contenido clinico. Aunque alguien intente fijar otra cosa, `fijar` la rechaza.

Cada preferencia guarda su procedencia (explicita | inferida_confirmada) para
trazabilidad, y el perfil es legible y corregible por el propio profesional.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Lista blanca: clave -> conjunto de valores permitidos (None = texto libre breve).
PREFERENCIAS_PERMITIDAS: dict[str, set | None] = {
    "verbosidad": {"conciso", "normal", "detallado"},
    "mostrar_grado_evidencia": {True, False},
    "listar_opciones_descartadas": {True, False},
    "formato_cita": {"inline", "pie", "compacto"},
    "tono": {"directo", "didactico"},
    "idioma": {"es", "en", "ca", "gl", "eu"},
    "subespecialidad": None,  # p. ej. "enfermedades infecciosas" (texto libre breve)
}

# Texto que explica, en el system prompt, que estas preferencias son subordinadas.
PREAMBULO_SEGURIDAD = (
    "Las siguientes son preferencias de PRESENTACION del profesional. Ajustan el "
    "formato y el detalle de la respuesta, pero NUNCA anulan los guardarrailes: la "
    "respuesta sigue obligada a citar la fuente, a abstenerse si la guia no lo cubre "
    "y a activar la supervision medica (HITL) cuando proceda. No alteran el contenido "
    "clinico."
)


@dataclass
class Preferencia:
    valor: Any
    fuente: str          # "explicita" | "inferida_confirmada"
    actualizado: str     # ISO-8601


@dataclass
class UserProfile:
    user_id: str
    preferencias: dict[str, Preferencia] = field(default_factory=dict)

    def fijar(self, clave: str, valor: Any, fuente: str = "explicita",
              ts: str | None = None) -> None:
        if clave not in PREFERENCIAS_PERMITIDAS:
            raise ValueError(f"Preferencia no permitida: {clave!r}. Solo forma, "
                             f"nunca contenido clinico ni guardarrailes.")
        permitidos = PREFERENCIAS_PERMITIDAS[clave]
        if permitidos is not None and valor not in permitidos:
            raise ValueError(f"Valor {valor!r} no permitido para {clave!r}; "
                             f"admitidos: {sorted(map(str, permitidos))}")
        if fuente not in {"explicita", "inferida_confirmada"}:
            raise ValueError("fuente debe ser 'explicita' o 'inferida_confirmada'")
        self.preferencias[clave] = Preferencia(
            valor=valor, fuente=fuente, actualizado=ts or datetime.now().isoformat())

    def quitar(self, clave: str) -> None:
        self.preferencias.pop(clave, None)

    def valor(self, clave: str, por_defecto: Any = None) -> Any:
        p = self.preferencias.get(clave)
        return p.valor if p else por_defecto

    # --- Render como contexto para el system prompt (subordinado a seguridad) ---
    def como_contexto(self) -> str:
        if not self.preferencias:
            return ""
        lineas = [PREAMBULO_SEGURIDAD, "", "Preferencias del profesional:"]
        etiquetas = {
            "verbosidad": "Nivel de detalle",
            "mostrar_grado_evidencia": "Mostrar siempre el grado de evidencia",
            "listar_opciones_descartadas": "Listar siempre las opciones descartadas",
            "formato_cita": "Formato de cita",
            "tono": "Tono",
            "idioma": "Idioma",
            "subespecialidad": "Subespecialidad",
        }
        for clave, pref in sorted(self.preferencias.items()):
            lineas.append(f"  - {etiquetas.get(clave, clave)}: {pref.valor}")
        return "\n".join(lineas)

    # --- Serializacion ---
    def to_dict(self) -> dict:
        return {"user_id": self.user_id,
                "preferencias": {k: asdict(v) for k, v in self.preferencias.items()}}

    @staticmethod
    def from_dict(d: dict) -> "UserProfile":
        prof = UserProfile(user_id=d["user_id"])
        for k, v in d.get("preferencias", {}).items():
            prof.preferencias[k] = Preferencia(**v)
        return prof


class ProfileStore:
    """Un fichero JSON por usuario en {memory}/profiles/{user_id}.json."""

    def __init__(self, dir_perfiles: Path):
        self.dir = dir_perfiles
        self.dir.mkdir(parents=True, exist_ok=True)

    def _ruta(self, user_id: str) -> Path:
        seguro = "".join(c for c in user_id if c.isalnum() or c in "-_")
        return self.dir / f"{seguro}.json"

    def cargar(self, user_id: str) -> UserProfile:
        ruta = self._ruta(user_id)
        if ruta.exists():
            return UserProfile.from_dict(json.loads(ruta.read_text(encoding="utf-8")))
        return UserProfile(user_id=user_id)

    def guardar(self, perfil: UserProfile) -> None:
        self._ruta(perfil.user_id).write_text(
            json.dumps(perfil.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
