import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Cita, Mensaje } from "../types";

const NODO_LABEL: Record<string, string> = {
  intake: "Recibiendo", router: "Clasificando", ecl: "Análisis de clase (SNOMED)",
  recuperar: "Recuperando guías", recuperar_vectorial: "Recuperando (vectorial)",
  recuperar_grafo: "Recuperando (grafo)", juez: "Fusionando evidencia",
  suficiencia: "Comprobando evidencia", sintetizar: "Redactando", verificar: "Verificando fidelidad",
  recuperar_dirigido: "Buscando respaldo adicional",
  hitl: "Requiere confirmación", abstener: "Sin respaldo", finalizar: "Listo", audit: "Registrando",
};

export default function Message({
  m, onCite, onHitl,
}: {
  m: Mensaje;
  onCite: (c: Cita[]) => void;
  onHitl: (texto: string) => void;
}) {
  const [copiado, setCopiado] = useState(false);
  const [hitlTexto, setHitlTexto] = useState("");

  if (m.role === "medico") {
    return (
      <div className="msg">
        <div className="who">Tú</div>
        <div className="bubble-medico">{m.content}</div>
      </div>
    );
  }

  const copiar = async () => {
    await navigator.clipboard.writeText(m.content);
    setCopiado(true); setTimeout(() => setCopiado(false), 1500);
  };

  return (
    <div className="msg">
      <div className="who">Asistente {m.model ? `· ${m.model}` : ""}</div>

      {m.streaming && m.pasos && m.pasos.length > 0 && !m.content && (
        <div className="trace">
          <span className="dot" />
          {m.pasos.map((p, i) => (
            <span key={i} className="step">{NODO_LABEL[p] || p}{i < m.pasos!.length - 1 ? " →" : "…"}</span>
          ))}
        </div>
      )}

      {m.content && (
        <div className="bubble-asistente">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
        </div>
      )}

      {m.hitl && m.hitl.tipo === "corazonamiento" && (
        <div className="hitl corazonamiento">
          {m.hitl.razonamiento && <p style={{ fontSize: 13, marginTop: 0 }}>🧠 {m.hitl.razonamiento}</p>}
          <h4>{m.hitl.pregunta_al_medico}</h4>
          {m.hitl.opciones && m.hitl.opciones.length > 0 && (
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
              {m.hitl.opciones.map((op: string, i: number) => (
                <button key={i} className="btn" style={{ width: "auto" }} onClick={() => onHitl(op)}>
                  {op}
                </button>
              ))}
            </div>
          )}
          <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
            <input className="field" style={{ flex: 1, background: "var(--panel)", border: "1px solid var(--border)", color: "var(--text)", borderRadius: 8, padding: 9 }}
                   placeholder="…o escribe tu respuesta" value={hitlTexto}
                   onChange={(e) => setHitlTexto(e.target.value)} />
            <button className="btn" style={{ width: "auto" }}
                    onClick={() => { if (hitlTexto.trim()) onHitl(hitlTexto.trim()); }}>
              Enviar
            </button>
          </div>
          <button className="btn ghost" style={{ width: "auto", marginTop: 8 }}
                  onClick={() => onHitl("Responde con lo que haya, sin más preguntas.")}>
            Responde con lo que hay
          </button>
        </div>
      )}

      {m.hitl && m.hitl.tipo !== "corazonamiento" && (
        <div className="hitl">
          <h4>⚠️ Se requiere tu confirmación ({m.hitl.motivo})</h4>
          <p style={{ fontSize: 13 }}>{m.hitl.pregunta_al_medico}</p>
          <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
            <input className="field" style={{ flex: 1, background: "var(--panel)", border: "1px solid var(--border)", color: "var(--text)", borderRadius: 8, padding: 9 }}
                   placeholder="Aporta el dato o confirma…" value={hitlTexto}
                   onChange={(e) => setHitlTexto(e.target.value)} />
            <button className="btn" style={{ width: "auto" }}
                    onClick={() => { if (hitlTexto.trim()) onHitl(hitlTexto.trim()); }}>
              Continuar
            </button>
          </div>
        </div>
      )}

      {m.content && !m.streaming && (
        <div className="msg-actions">
          <button className="copy-btn" onClick={copiar}>{copiado ? "✓ Copiado" : "Copiar"}</button>
          {m.citations && m.citations.length > 0 && (
            <button className="chip" onClick={() => onCite(m.citations!)}>
              📑 {m.citations.length} fuentes
            </button>
          )}
          {m.veredicto && (
            <span className={"veredicto " + m.veredicto}>fidelidad: {m.veredicto}</span>
          )}
          {m.abstenida && <span className="muted" style={{ fontSize: 11 }}>abstención</span>}
        </div>
      )}

      {m.content && !m.streaming && (m.ruta?.length || m.detalle) && (
        <details style={{ marginTop: 8, fontSize: 13, opacity: 0.92 }}>
          <summary style={{ cursor: "pointer", userSelect: "none" }}>
            🧠 Proceso de razonamiento ({m.ruta?.length || 0} pasos)
          </summary>
          <div style={{ marginTop: 8, padding: "10px 12px", borderRadius: 8,
                        background: "rgba(127,127,127,.08)", border: "1px solid rgba(127,127,127,.25)",
                        display: "flex", flexDirection: "column", gap: 10 }}>

            {m.ruta && m.ruta.length > 0 && (
              <div>
                <strong>Ruta ejecutada</strong>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                  {m.ruta.map((p, i) => {
                    const base = p.split("(")[0].replace(/\+.*$/, "");
                    const label = NODO_LABEL[base];
                    return (
                      <span key={i} title={label || undefined}
                            style={{ fontFamily: "monospace", fontSize: 12, padding: "2px 7px",
                                     borderRadius: 4, background: "rgba(127,127,127,.15)" }}>
                        {p}{i < m.ruta!.length - 1 ? " →" : ""}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}

            {m.detalle?.clasificacion && Object.keys(m.detalle.clasificacion).length > 0 && (
              <div>
                <strong>Clasificación del router</strong>
                <div style={{ marginTop: 4 }}>
                  tipo: <code>{m.detalle.clasificacion.tipo_consulta}</code> ·
                  nivel: <code>{m.detalle.clasificacion.nivel}</code> ·
                  estrategia: <code>{m.detalle.clasificacion.estrategia_recuperacion}</code> ·
                  multi-aspecto: <code>{String(m.detalle.clasificacion.multi_aspecto)}</code> ·
                  lógica numérica: <code>{String(m.detalle.clasificacion.logica_numerica)}</code> ·
                  decisión seria: <code>{String(m.detalle.clasificacion.decision_seria)}</code>
                  {m.detalle.clasificacion.guias_candidatas?.length > 0 && (
                    <> · guías candidatas: <code>{m.detalle.clasificacion.guias_candidatas.join(", ")}</code></>
                  )}
                  {m.detalle.clasificacion.datos_faltantes?.length > 0 && (
                    <> · datos faltantes: <code>{m.detalle.clasificacion.datos_faltantes.join(", ")}</code></>
                  )}
                </div>
              </div>
            )}

            {m.detalle?.ecl && (
              <div>
                <strong>Análisis de clase (SNOMED/ECL)</strong>
                <div style={{ marginTop: 4 }}>
                  clase: <code>{m.detalle.ecl.clase}</code>
                  {m.detalle.ecl.poblacion && <> · población: <code>{m.detalle.ecl.poblacion}</code></>}
                  {" "}· miembros en el corpus: <code>{m.detalle.ecl.n_miembros_en_corpus}</code>
                </div>
              </div>
            )}

            {m.detalle?.suficiencia?.accion && (
              <div>
                <strong>Suficiencia de la evidencia (CRAG)</strong>
                <div style={{ marginTop: 4 }}>
                  acción: <code>{m.detalle.suficiencia.accion}</code>
                  {m.detalle.suficiencia.motivo && <> — {m.detalle.suficiencia.motivo}</>}
                </div>
              </div>
            )}

            {m.detalle?.veredicto && Object.keys(m.detalle.veredicto).length > 0 && (
              <div>
                <strong>Verificación de fidelidad</strong>
                <div style={{ marginTop: 4 }}>
                  veredicto: <code>{m.detalle.veredicto.veredicto || "—"}</code>
                  {m.detalle.veredicto.explicacion && <div style={{ marginTop: 3 }}>{m.detalle.veredicto.explicacion}</div>}
                  {m.detalle.veredicto.afirmaciones_sin_respaldo?.length > 0 && (
                    <div style={{ marginTop: 4 }}>
                      ⚠ afirmaciones sin respaldo detectadas:
                      <ul style={{ margin: "4px 0 0 0", paddingLeft: 18 }}>
                        {m.detalle.veredicto.afirmaciones_sin_respaldo.map((a: string, i: number) => <li key={i}>{a}</li>)}
                      </ul>
                    </div>
                  )}
                  {m.detalle.veredicto.lentes?.length > 0 && (
                    <div style={{ marginTop: 4 }}>
                      🔍 lentes (deóntica/numérica):
                      <ul style={{ margin: "4px 0 0 0", paddingLeft: 18 }}>
                        {m.detalle.veredicto.lentes.map((l: string, i: number) => <li key={i}>{l}</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="muted" style={{ fontSize: 11 }}>
              La traza de grafo por cita (semilla → aserción → fragmento) está en «📑 fuentes».
            </div>
          </div>
        </details>
      )}
    </div>
  );
}
