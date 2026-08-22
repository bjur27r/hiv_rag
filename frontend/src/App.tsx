import { useEffect, useState } from "react";
import { api } from "./api";
import Login from "./components/Login";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";
import Settings from "./components/Settings";
import type { Mensaje, Proveedor, Thread, Usuario } from "./types";
import CurationPanel from "./components/CurationPanel";

export default function App() {
  const [user, setUser] = useState<Usuario | null | undefined>(undefined);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Mensaje[]>([]);
  const [providers, setProviders] = useState<Proveedor[]>([]);
  const [sel, setSel] = useState({ provider: "anthropic", modelo: "" });
  const [modo, setModo] = useState<"directa" | "corazonamiento">("directa");
  const [busy, setBusy] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [view, setView] = useState<"chat" | "curation">("chat");

  useEffect(() => { api.me().then(setUser).catch(() => setUser(null)); }, []);

  useEffect(() => {
    if (!user) return;
    refreshThreads();
    api.models().then((d) => {
      setProviders(d.proveedores);
      const first = d.proveedores.find((p: Proveedor) => p.disponible) || d.proveedores[0];
      if (first) setSel({ provider: first.provider, modelo: first.modelos[0]?.id || "" });
    });
  }, [user]);

  async function refreshThreads() { setThreads(await api.threads()); }

  async function openThread(id: string) {
    setCurrentId(id);
    setView("chat");
    const t = await api.thread(id);
    setMessages(t.messages);
  }
  function newThread() { setCurrentId(null); setMessages([]); setView("chat"); }
  async function deleteThread(id: string) {
    await api.deleteThread(id);
    if (id === currentId) newThread();
    refreshThreads();
  }

  async function ensureThread(): Promise<string> {
    if (currentId) return currentId;
    const t = await api.createThread();
    setCurrentId(t.id);
    refreshThreads();
    return t.id;
  }

  async function runAsk(threadId: string, body: any) {
    setMessages((prev) => [...prev, { role: "asistente", content: "", pasos: [], streaming: true }]);
    setBusy(true);
    try {
      await api.ask(threadId, body, (ev) => {
        setMessages((prev) => {
          const arr = [...prev];
          const last = { ...arr[arr.length - 1] };
          if (ev.type === "paso") last.pasos = [...(last.pasos || []), ev.nodo];
          else if (ev.type === "respuesta") {
            last.content = ev.texto; last.citations = ev.citas; last.veredicto = ev.veredicto;
            last.abstenida = ev.abstenida; last.ruta = ev.ruta; last.detalle = ev.detalle;
            last.streaming = false; last.hitl = undefined;
          } else if (ev.type === "hitl") {
            last.hitl = ev.payload; last.streaming = false;
          } else if (ev.type === "error") {
            last.content = "⚠️ " + ev.mensaje; last.streaming = false;
          }
          arr[arr.length - 1] = last;
          return arr;
        });
      });
    } finally {
      setBusy(false);
      refreshThreads();
    }
  }

  async function send(texto: string) {
    const id = await ensureThread();
    setMessages((prev) => [...prev, { role: "medico", content: texto }]);
    await runAsk(id, { pregunta: texto, provider: sel.provider, modelo: sel.modelo, modo });
  }

  async function hitl(texto: string) {
    if (!currentId) return;
    setMessages((prev) => prev.map((m) => ({ ...m, hitl: undefined }))
      .concat({ role: "medico", content: texto }));
    await runAsk(currentId, { pregunta: "", resume: texto, provider: sel.provider, modelo: sel.modelo, modo });
  }

  if (user === undefined)
    return <div className="login-wrap"><div className="muted">Cargando…</div></div>;
  if (!user) return <Login onLogin={(u) => setUser(u)} />;

  return (
    <div className="layout">
      <Sidebar threads={threads} currentId={currentId} user={user}
               onNew={newThread} onOpen={openThread} onDelete={deleteThread}
               onSettings={() => setShowSettings(true)}
               onCuration={() => setView("curation")}
               onLogout={async () => {
                 await api.logout();
                 setUser(null); setThreads([]); setMessages([]); setCurrentId(null);
               }} />
      {view === "chat" ? (
        <Chat messages={messages} providers={providers} sel={sel} setSel={setSel}
              modo={modo} setModo={setModo}
              onSend={send} onHitl={hitl} busy={busy} />
      ) : (
        <div style={{ flex: 1, overflowY: "auto", background: "#fff" }}>
          <CurationPanel />
        </div>
      )}
      {showSettings && <Settings onClose={() => setShowSettings(false)} />}
    </div>
  );
}
