"""Subsistema de memoria y persistencia.

Dos capas, deliberadamente separadas:

  Capa 1 - Episodica (episodic.py): historial crudo, append-only, SIEMPRE
           seudonimizado. Dato de salud regulado. Sustrato de auditoria,
           evaluacion y mineria de huecos. NO se inyecta entero en los prompts.

  Capa 2 - Perfil destilado (profile.py): preferencias de FORMA del usuario,
           estructuradas, revisables y editables. Es lo que se inyecta como
           contexto. Por construccion (lista blanca) no puede contener ninguna
           preferencia que desactive un guardarrail ni que altere el contenido
           clinico.

Principio rector del subsistema: se personaliza el COMO (presentacion), nunca
el QUE clinico; y la personalizacion esta subordinada a la envolvente de
seguridad (cita, abstencion, HITL). El contexto de paciente vive en el episodio,
nunca en un perfil persistente.
"""

from .store import MemoryStore

__all__ = ["MemoryStore"]
