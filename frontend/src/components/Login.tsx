import { useState } from "react";
import type { FormEvent } from "react";
import { api } from "../api";
import type { Usuario } from "../types";

export default function Login({ onLogin }: { onLogin: (u: Usuario) => void }) {
  const [modo, setModo] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [cargando, setCargando] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError(""); setCargando(true);
    try {
      const r = modo === "login"
        ? await api.login(email, password)
        : await api.register(email, password);
      onLogin(r.user);
    } catch (err: any) {
      setError(err.message || "Error");
    } finally {
      setCargando(false);
    }
  }

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <h1>Asistente Clínico VIH</h1>
        <p className="sub">Consulta sobre guías GeSIDA — respuestas citadas y verificadas</p>
        <div className="field">
          <label>Email</label>
          <input type="email" value={email} required autoFocus
                 onChange={(e) => setEmail(e.target.value)} />
        </div>
        <div className="field">
          <label>Contraseña</label>
          <input type="password" value={password} required minLength={6}
                 onChange={(e) => setPassword(e.target.value)} />
        </div>
        {error && <div className="error">{error}</div>}
        <button className="btn" disabled={cargando}>
          {cargando ? "…" : modo === "login" ? "Entrar" : "Crear cuenta"}
        </button>
        <div style={{ marginTop: 14, textAlign: "center" }}>
          <button type="button" className="link"
                  onClick={() => { setModo(modo === "login" ? "register" : "login"); setError(""); }}>
            {modo === "login" ? "¿No tienes cuenta? Regístrate" : "Ya tengo cuenta — entrar"}
          </button>
        </div>
      </form>
    </div>
  );
}
