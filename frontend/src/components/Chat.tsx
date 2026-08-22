import { useEffect, useRef, useState } from "react";
import Message from "./Message";
import type { Cita, Mensaje, Proveedor } from "../types";

export default function Chat({
  messages, providers, sel, setSel, modo, setModo, onSend, onHitl, busy,
}: {
  messages: Mensaje[];
  providers: Proveedor[];
  sel: { provider: string; modelo: string };
  setSel: (s: { provider: string; modelo: string }) => void;
  modo: "directa" | "corazonamiento";
  setModo: (m: "directa" | "corazonamiento") => void;
  onSend: (texto: string) => void;
  onHitl: (texto: string) => void;
  busy: boolean;
}) {
  const [texto, setTexto] = useState("");
  const [citas, setCitas] = useState<Cita[] | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const prov = providers.find((p) => p.provider === sel.provider);

  function enviar() {
    const t = texto.trim();
    if (!t || busy) return;
    setTexto("");
    onSend(t);
  }

  return (
    <div className="chat">
      <div className="topbar">
        <div className="modo-toggle" role="group" aria-label="Modo de reflexión">
          <button
            className={"modo-btn" + (modo === "directa" ? " activo" : "")}
            onClick={() => setModo("directa")}
            title="Pregunta puntual → respuesta citada directa.">
            Consulta directa
          </button>
          <button
            className={"modo-btn" + (modo === "corazonamiento" ? " activo" : "")}
            onClick={() => setModo("corazonamiento")}
            title="El asistente razona contigo y te consulta cuando la guía depende de un dato que decides tú.">
            🧠 Razonar el caso
          </button>
        </div>
        <div className="spacer" />
        <span className="muted" style={{ fontSize: 12 }}>Modelo:</span>
        <select value={sel.provider}
                onChange={(e) => {
                  const p = providers.find((x) => x.provider === e.target.value);
                  setSel({ provider: e.target.value, modelo: p?.modelos[0]?.id || "" });
                }}>
          {providers.map((p) => (
            <option key={p.provider} value={p.provider}>
              {p.label}{p.disponible ? "" : " (sin clave)"}
            </option>
          ))}
        </select>
        <select value={sel.modelo} onChange={(e) => setSel({ ...sel, modelo: e.target.value })}>
          {prov?.modelos.map((mo) => <option key={mo.id} value={mo.id}>{mo.label}</option>)}
        </select>
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div className="msg muted" style={{ textAlign: "center", marginTop: 60 }}>
            Formula tu consulta sobre las guías GeSIDA de VIH.
          </div>
        )}
        {messages.map((m, i) => (
          <Message key={m.id ?? `live-${i}`} m={m} onCite={setCitas} onHitl={onHitl} />
        ))}
        <div ref={endRef} />
      </div>

      <div className="composer">
        <div className="row">
          <textarea rows={1} value={texto} placeholder="Escribe tu pregunta clínica…"
                    onChange={(e) => setTexto(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); enviar(); } }} />
          <button className="send" onClick={enviar} disabled={busy}>{busy ? "…" : "Enviar"}</button>
        </div>
        <div className="disclaimer">
          Apoyo a la decisión del médico. No prescribe. Cada afirmación se ancla a la guía y se verifica.
        </div>
      </div>

      {citas && (
        <div className="cite-panel">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <h3>Trazabilidad — fuentes citadas</h3>
            <button className="btn ghost" style={{ width: "auto" }} onClick={() => setCitas(null)}>✕</button>
          </div>
          {citas.map((c) => (
            <div key={c.n} className="cite-frag">
              <div className="meta">
                [{c.n}] {c.guia} · {c.seccion} · pág. {c.pagina}
                {c.vigencia && (
                  <span title="Fecha de vigencia de la guía citada"
                        style={{ marginLeft: 8, padding: "1px 6px", borderRadius: 4,
                                 fontSize: "0.85em", background: "rgba(120,120,120,.18)" }}>
                    vigencia {c.vigencia}
                  </span>
                )}
              </div>
              {c.reasoning_trace && c.reasoning_trace.length > 0 && (
                <div className="trace" style={{ fontSize: "12px", color: "#666", marginBottom: "8px", backgroundColor: "#f0f4f8", padding: "6px", borderRadius: "4px" }}>
                  <strong>Traza CatRAG:</strong>
                  <ul style={{ margin: "4px 0 0 0", paddingLeft: "20px" }}>
                    {c.reasoning_trace.map((tr, idx) => <li key={idx}>{tr}</li>)}
                  </ul>
                </div>
              )}
              <div>{c.texto}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
