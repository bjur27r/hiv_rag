"""Modo 'Razonar el caso' — bucle de CO-RAZONAMIENTO dirigido por el modelo.

Version MINIMA (estilo Anthropic): el MODELO decide el flujo eligiendo una ACCION
estructurada por turno —buscar / preguntar / responder— en vez de un grafo de
estados explicito. Hay dos "herramientas" y una salida final:

  - buscar      -> recupera fragmentos de las guias (el retriever existente).
  - preguntar   -> consulta al medico cuando la guia se BIFURCA en un criterio
                   que no consta Y eso cambia la recomendacion (= el HITL: pausa).
  - responder   -> redacta la respuesta CITADA y la pasa por el verificador.

Reusa, sin duplicar: el retriever (via graph._retrievers), generate.generar_respuesta
y verify.verificar_fidelidad (el verificador sigue siendo la barrera anti-alucinacion).

Es ADITIVO: NO toca el modo 'Consulta directa' (graph.py). Camino de codigo
separado, que el backend selecciona por 'modo'. Corre en SIMULADO sin claves
(heuristica determinista) para ejercitar el flujo sin gasto.

La decision de preguntar vive aqui en el razonamiento del modelo (bucle), no en
codigo Python. Si los evals muestran sobre-pregunta o falta de trazabilidad, el
paso siguiente es endurecer estos puntos con el nodo detector + post-filtro del
diseno completo (ver memoria del proyecto).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..llm.client import (LLM, MODELO_ROUTER, MODELO_SINTESIS, MODELO_VERIFICADOR, _DEFAULTS)
from ..llm.generate import generar_respuesta
from ..llm.verify import verificar_fidelidad

# Topes anti-pesadez / anti-bucle (hermanos de MAX_REC/MAX_GEN del grafo).
MAX_ITERS = 6       # pasos internos totales (busquedas + decisiones)
MAX_PREGUNTAS = 3   # cuantas veces, como mucho, se consulta al medico


# --------------------------------------------------------------------- estado
@dataclass
class Sesion:
    """Estado de una consulta en modo co-razonamiento (pausa/reanuda entre turnos)."""
    pregunta: str
    user_id: str = "anon"
    thread_id: str = "consulta"
    cfg_llm: dict = field(default_factory=dict)
    simulado: bool | None = None          # fuerza modo simulado (None = auto por clave)
    aportes: list[str] = field(default_factory=list)   # lo que el medico va aportando
    hits: list = field(default_factory=list)           # list[Hit] de la ultima recuperacion
    iter: int = 0
    preguntas_hechas: int = 0
    traza: list[str] = field(default_factory=list)     # auditoria del recorrido


def _llm(cfg: dict, rol: str, simulado: bool | None) -> LLM:
    """LLM del rol segun la seleccion de la peticion (mismo helper que en graph.py)."""
    c = cfg or {}
    prov = c.get("provider")
    modelo = c.get(rol)
    if prov and not modelo:
        modelo = _DEFAULTS.get(prov, {}).get(rol)
    if not modelo:
        modelo = {"router": MODELO_ROUTER, "sintesis": MODELO_SINTESIS,
                  "verificador": MODELO_VERIFICADOR}[rol]
    return LLM(modelo, simulado=simulado, provider=prov)


# ----------------------------------------------------- decision (el "cerebro")
SYSTEM_DECISION = """Eres el modulo de RAZONAMIENTO ESTRATEGICO de un asistente clinico \
sobre las guias GeSIDA de VIH, en modo CO-RAZONAMIENTO con el medico. NO redactas aqui la \
respuesta clinica: decides la SIGUIENTE ACCION.

Acciones posibles:
- buscar: recuperar fragmentos de las guias. Indica 'consulta_busqueda'.
- preguntar: consultar al medico SOLO cuando la guia se BIFURCA segun un criterio que el \
medico NO ha aportado Y esa bifurcacion CAMBIA la recomendacion. Aporta 'razonamiento' \
(breve, lo que la guia distingue), 'criterio' y 'opciones' (las ramas, tomadas de los \
fragmentos recuperados).
- responder: cuando ya tienes base suficiente para una respuesta citada.

Reglas innegociables:
1. Solo te apoyas en lo recuperado; no aportas conocimiento clinico propio. Si aun no has \
buscado, la primera accion DEBE ser buscar.
2. preguntar SOLO si la rama cambia la recomendacion y el criterio no consta. Si las ramas \
convergen, o es una pregunta general 'de libro', NO preguntes: responde.
3. No preguntes de mas. Si el medico ya aporto lo que zanja el caso, responde.
4. Las 'opciones' deben ser ramas presentes en los fragmentos, no inventadas.

Rellena todos los campos del esquema; deja vacios ('' o []) los que no apliquen a la accion."""

ACCION_SCHEMA = {
    "type": "object",
    "properties": {
        "accion": {"type": "string", "enum": ["buscar", "preguntar", "responder"]},
        "consulta_busqueda": {"type": "string"},
        "razonamiento": {"type": "string"},
        "criterio": {"type": "string"},
        "opciones": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["accion", "consulta_busqueda", "razonamiento", "criterio", "opciones"],
    "additionalProperties": False,
}

# Heuristica de simulado: dispara una bifurcacion cuando la pregunta huele a
# parametro de paciente (FG/CD4/renal...) y aun no consta.
_RX_PARAM = re.compile(r"\b(fg|filtrado|aclaramiento|cd4|carga viral|copias|renal|"
                       r"insuficiencia|genotipo|resistenc|hbsag|hepatitis b|embaraz)\b", re.I)


def _accion_heuristica(sesion: Sesion) -> dict:
    base = {"accion": "responder", "consulta_busqueda": "", "razonamiento": "",
            "criterio": "", "opciones": []}
    if not sesion.hits:
        return {**base, "accion": "buscar", "consulta_busqueda": sesion.pregunta}
    if (_RX_PARAM.search(sesion.pregunta) and not sesion.aportes
            and sesion.preguntas_hechas < MAX_PREGUNTAS):
        return {**base, "accion": "preguntar",
                "razonamiento": "La guia ajusta la recomendacion segun un parametro del "
                                "paciente que no consta en la consulta.",
                "criterio": "parametro clinico que bifurca la recomendacion (p. ej. FG/CD4/genotipo)",
                "opciones": ["Por debajo del umbral", "Por encima del umbral", "No consta / no lo se"]}
    return base


def _resumen_hits(hits: list, maxn: int = 8) -> str:
    lineas = []
    for i, h in enumerate(hits[:maxn], start=1):
        lineas.append(f"[{i}] (guia={h.guia}, seccion={h.seccion[:50]}, pag={h.pagina})\n{h.texto[:500]}")
    return "\n\n".join(lineas)


def _user_decision(sesion: Sesion) -> str:
    ctx = _resumen_hits(sesion.hits) if sesion.hits else "(aun no has recuperado nada)"
    aportes = "\n".join(f"- {a}" for a in sesion.aportes) or "(ninguno)"
    return (f"Pregunta del medico:\n{sesion.pregunta}\n\n"
            f"Datos que el medico ya ha aportado:\n{aportes}\n\n"
            f"Fragmentos recuperados hasta ahora:\n{ctx}\n\n"
            f"Preguntas ya hechas al medico: {sesion.preguntas_hechas} (max {MAX_PREGUNTAS}).\n"
            f"Elige la siguiente accion.")


def _decidir(sesion: Sesion) -> dict:
    llm = _llm(sesion.cfg_llm, "router", sesion.simulado)   # decisor = modelo barato (Haiku/mini)
    return llm.json(SYSTEM_DECISION, _user_decision(sesion), ACCION_SCHEMA,
                    max_tokens=600, simulado_valor=_accion_heuristica(sesion))


# --------------------------------------------------------------- herramientas
_RETR = None


def _buscar(sesion: Sesion, consulta: str) -> None:
    """Herramienta 'buscar_guia': recupera y deja los hits en la sesion."""
    global _RETR
    if _RETR is None:
        from .graph import _retrievers
        base, decomp, catrag = _retrievers()
        _RETR = {"base": base, "catrag": catrag}
        
    if _RETR.get("catrag"):
        sesion.hits = _RETR["catrag"].search(consulta, k=8)
    else:
        sesion.hits = _RETR["base"].search(consulta, k=8)


def _citas(hits: list) -> list[dict]:
    out = []
    for i, h in enumerate(hits[:8], start=1):
        out.append({"n": i, "chunk_id": getattr(h, "chunk_id", ""), "guia": getattr(h, "guia", ""), "seccion": getattr(h, "seccion", ""),
                    "pagina": getattr(h, "pagina", ""), "texto": (getattr(h, "texto", "") or "")[:1200],
                    "vigencia": getattr(h, "fecha_vigencia", ""),
                    "reasoning_trace": getattr(h, "reasoning_trace", [])})
    return out


def _responder(sesion: Sesion) -> dict:
    """Salida final: redacta citado (sintesis) y verifica fidelidad (barrera)."""
    if not sesion.hits:
        return {"tipo": "abstener", "traza": sesion.traza,
                "texto": "Las guias proporcionadas no permiten una respuesta fiable a esta consulta."}
    extra = (f"\n[Contexto aportado por el medico: {' ; '.join(sesion.aportes)}]"
             if sesion.aportes else "")
    gen = generar_respuesta(sesion.pregunta + extra, sesion.hits,
                            llm=_llm(sesion.cfg_llm, "sintesis", sesion.simulado))
    v = verificar_fidelidad(sesion.pregunta, gen["respuesta"], sesion.hits,
                            llm=_llm(sesion.cfg_llm, "verificador", sesion.simulado))
    return {"tipo": "respuesta", "texto": gen["respuesta"], "abstenida": gen["abstenida"],
            "veredicto": v.get("veredicto"), "citas": _citas(sesion.hits), "traza": sesion.traza}


# -------------------------------------------------------------- bucle (motor)
def avanzar(sesion: Sesion) -> dict:
    """Corre pasos internos hasta que toque PREGUNTAR al medico o haya RESPUESTA.

    Devuelve un evento:
      {'tipo':'preguntar', razonamiento, criterio, opciones}  -> pausa (espera al medico)
      {'tipo':'respuesta', texto, veredicto, citas, traza}
      {'tipo':'abstener',  texto, traza}
    """
    while sesion.iter < MAX_ITERS:
        dec = _decidir(sesion)
        accion = dec.get("accion")

        if accion == "buscar":
            q = dec.get("consulta_busqueda") or _q_con_aportes(sesion)
            _buscar(sesion, q)
            sesion.iter += 1
            sesion.traza.append(f"buscar('{q[:40]}') -> {len(sesion.hits)} hits")
            continue

        if accion == "preguntar" and sesion.hits and sesion.preguntas_hechas < MAX_PREGUNTAS:
            sesion.preguntas_hechas += 1
            sesion.iter += 1
            crit = dec.get("criterio", "")
            sesion.traza.append(f"preguntar('{crit[:30]}')")
            return {"tipo": "preguntar", "razonamiento": dec.get("razonamiento", ""),
                    "criterio": crit, "opciones": dec.get("opciones", [])}

        # responder (o se agotaron las preguntas / no habia que preguntar)
        sesion.traza.append("responder")
        return _responder(sesion)

    sesion.traza.append("responder(tope iteraciones)")
    return _responder(sesion)


def _q_con_aportes(sesion: Sesion) -> str:
    if not sesion.aportes:
        return sesion.pregunta
    return f"{sesion.pregunta} [contexto del medico: {' ; '.join(sesion.aportes)}]"


# ----------------------------------------------- API de sesion (pausa/reanuda)
# Almacen EN MEMORIA por hilo (como MemorySaver del grafo). Produccion: persistir.
_SESIONES: dict[str, Sesion] = {}


def iniciar(pregunta: str, user_id: str = "anon", thread_id: str = "consulta",
            cfg_llm: dict | None = None, simulado: bool | None = None) -> dict:
    """Arranca una consulta en modo co-razonamiento. Devuelve el primer evento."""
    pregunta_final = pregunta
    try:
        from ..memory import MemoryStore
        mem = MemoryStore()
        contexto = mem.construir_contexto_sistema(user_id, thread_id, pregunta)
        mem.registrar_turno(thread_id, user_id, "medico", pregunta)
        
        if contexto and "Contexto de esta conversacion" in contexto:
            llm_router = _llm(cfg_llm, "router", simulado)
            prompt_rew = (f"Reescribe la ultima pregunta del usuario para que sea autocontenida "
                          f"y se entienda sin leer el historial. Incorpora sujetos, verbos o contextos "
                          f"omitidos. Responde SOLO con la pregunta reescrita.\n\n"
                          f"Historial:\n{contexto}\n\nPregunta actual: {pregunta_final}\n\nPregunta reescrita:")
            res = llm_router.texto("Eres un asistente util.", prompt_rew, max_tokens=150)
            if res and len(res) > 5:
                pregunta_final = res.strip().strip('"').strip("'")
    except Exception as e:
        print(f"Error reescribiendo query en corazonamiento: {e}")

    s = Sesion(pregunta=pregunta_final, user_id=user_id, thread_id=thread_id,
               cfg_llm=cfg_llm or {}, simulado=simulado)
    _SESIONES[thread_id] = s
    return avanzar(s)


def responder_medico(thread_id: str, respuesta: str) -> dict:
    """Reanuda una consulta pausada con el aporte del medico."""
    s = _SESIONES.get(thread_id)
    if s is None:
        return {"tipo": "error", "mensaje": "sesion de co-razonamiento no encontrada"}
    s.aportes.append(respuesta)
    s.traza.append(f"medico: {respuesta[:40]}")
    return avanzar(s)
