import type { SSEvento } from "./types";

const BASE = "/api";

async function j(path: string, opts: RequestInit = {}): Promise<any> {
  const res = await fetch(BASE + path, {
    credentials: "include",
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  me: () => j("/auth/me"),
  login: (email: string, password: string) =>
    j("/auth/login", { method: "POST", body: JSON.stringify({ email, password }) }),
  register: (email: string, password: string) =>
    j("/auth/register", { method: "POST", body: JSON.stringify({ email, password }) }),
  logout: () => j("/auth/logout", { method: "POST" }),
  changePassword: (actual: string, nueva: string) =>
    j("/auth/change-password", {
      method: "POST",
      body: JSON.stringify({ password_actual: actual, password_nueva: nueva }),
    }),
  models: () => j("/models"),
  threads: () => j("/threads"),
  createThread: () => j("/threads", { method: "POST" }),
  thread: (id: string) => j("/threads/" + id),
  deleteThread: (id: string) => j("/threads/" + id, { method: "DELETE" }),

  // Streaming SSE de /ask: invoca onEvent por cada evento del grafo.
  ask: async (
    threadId: string,
    body: { pregunta: string; provider?: string; modelo?: string; resume?: string; modo?: string },
    onEvent: (e: SSEvento) => void
  ): Promise<void> => {
    const res = await fetch(`${BASE}/threads/${threadId}/ask`, {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok || !res.body) throw new Error("Error al iniciar la consulta");
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx: number;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const bloque = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const linea = bloque.split("\n").find((l) => l.startsWith("data: "));
        if (linea) {
          try { onEvent(JSON.parse(linea.slice(6))); } catch { /* ignore */ }
        }
      }
    }
  },

  // -- Curation Golden Graph --
  
  getUncuratedEntities: () => j("/curation/entities"),
  getEntityDetails: (name: string) => j(`/curation/entity/${encodeURIComponent(name)}`),
  curateEntity: (name: string, decision: any) => j(`/curation/entity/${encodeURIComponent(name)}/curate`, { method: "POST", body: JSON.stringify(decision) }),
  manualSearchUmls: (term: string) => j(`/curation/manual_search?term=${encodeURIComponent(term)}`),
  searchGraphNodes: (q: string) => j(`/curation/search_nodes?q=${encodeURIComponent(q)}`),
  getGraphData: (query: string = "", hops: number = 2) => j(`/curation/graph_data?query=${encodeURIComponent(query)}&hops=${hops}`),
  getChunks: (limit: number = 100, offset: number = 0) => j(`/curation/chunks?limit=${limit}&offset=${offset}`),
  getChunkDetails: (chunkId: string) => j(`/curation/chunk/${encodeURIComponent(chunkId)}`),

  // -- Anclajes SNOMED (terminologia local) --
  getAnclajes: (estado: string = "pendientes", limit: number = 100, offset: number = 0, q: string = "") =>
    j(`/curation/anclajes?estado=${estado}&limit=${limit}&offset=${offset}&q=${encodeURIComponent(q)}`),
  decidirAnclaje: (body: { entidad: string; accion: string; sctid?: string; cui?: string; termino?: string }) =>
    j("/curation/anclajes/decidir", { method: "POST", body: JSON.stringify(body) }),
  linkerSearch: (term: string, slot: string = "") =>
    j(`/curation/linker_search?term=${encodeURIComponent(term)}${slot ? `&slot=${slot}` : ""}`)
};
