import type { Thread, Usuario } from "../types";

export default function Sidebar({
  threads, currentId, user, onNew, onOpen, onDelete, onSettings, onLogout, onCuration
}: {
  threads: Thread[];
  currentId: string | null;
  user: Usuario;
  onNew: () => void;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
  onSettings: () => void;
  onLogout: () => void;
  onCuration: () => void;
}) {
  return (
    <div className="sidebar">
      <div className="head">
        <div className="brand">Asistente VIH <small>Guías GeSIDA</small></div>
        <button className="btn" style={{ marginTop: 12 }} onClick={onNew}>+ Nueva consulta</button>
      </div>
      <div className="thread-list">
        {threads.length === 0 && <div className="muted" style={{ padding: 10, fontSize: 13 }}>Sin consultas todavía</div>}
        {threads.map((t) => (
          <div key={t.id}
               className={"thread-item" + (t.id === currentId ? " active" : "")}
               onClick={() => onOpen(t.id)}>
            <span className="thread-title">{t.title}</span>
            <span className="x" title="Borrar"
                  onClick={(e) => { e.stopPropagation(); onDelete(t.id); }}>✕</span>
          </div>
        ))}
      </div>
      <div className="foot">
        <div className="muted" style={{ marginBottom: 8, overflow: "hidden", textOverflow: "ellipsis" }}>
          {user.email}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button className="btn ghost" style={{ flex: 1 }} onClick={onCuration}>Curación</button>
          <button className="btn ghost" style={{ flex: 1 }} onClick={onSettings}>Ajustes</button>
          <button className="btn ghost" style={{ flex: 1 }} onClick={onLogout}>Salir</button>
        </div>
      </div>
    </div>
  );
}
