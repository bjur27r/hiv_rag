import { useState } from "react";
import type { FormEvent } from "react";
import { api } from "../api";

export default function Settings({ onClose }: { onClose: () => void }) {
  const [actual, setActual] = useState("");
  const [nueva, setNueva] = useState("");
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  async function cambiar(e: FormEvent) {
    e.preventDefault();
    setMsg(""); setErr("");
    try {
      await api.changePassword(actual, nueva);
      setMsg("Contraseña actualizada"); setActual(""); setNueva("");
    } catch (e: any) {
      setErr(e.message || "Error");
    }
  }

  return (
    <div className="modal-bg" onClick={onClose}>
      <form className="modal" onClick={(e) => e.stopPropagation()} onSubmit={cambiar}>
        <h3>Ajustes — cambiar contraseña</h3>
        <div className="field">
          <label>Contraseña actual</label>
          <input type="password" value={actual} required onChange={(e) => setActual(e.target.value)} />
        </div>
        <div className="field">
          <label>Nueva contraseña</label>
          <input type="password" value={nueva} required minLength={6} onChange={(e) => setNueva(e.target.value)} />
        </div>
        {msg && <div style={{ color: "var(--ok)", fontSize: 13 }}>{msg}</div>}
        {err && <div className="error">{err}</div>}
        <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
          <button type="button" className="btn secondary" onClick={onClose}>Cerrar</button>
          <button className="btn">Guardar</button>
        </div>
      </form>
    </div>
  );
}
