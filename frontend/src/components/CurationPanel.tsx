import { useState, useEffect } from "react";
import { api } from "../api";
import GraphExplorer from "./GraphExplorer";

const METODO_BADGE: Record<string, { bg: string; label: string }> = {
  juez: { bg: "#f8d7da", label: "juez LLM (revisar)" },
  fuzzy: { bg: "#fff3cd", label: "difuso" },
  exacto: { bg: "#d1e7dd", label: "exacto" },
  manual: { bg: "#cfe2ff", label: "manual" },
};

export default function CurationPanel() {
  const [activeTab, setActiveTab] = useState<"curation" | "explorer" | "chunks" | "anclajes">("explorer");

  // -- Estado para Curación --
  const [entities, setEntities] = useState<any[]>([]);
  const [selectedEntity, setSelectedEntity] = useState<any>(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [selectedSynonyms, setSelectedSynonyms] = useState<string[]>([]);
  const [umlsPropuestas, setUmlsPropuestas] = useState<any[]>([]);

  // -- Estado para Chunks --
  const [chunks, setChunks] = useState<any[]>([]);
  const [selectedChunk, setSelectedChunk] = useState<any>(null);

  // -- Estado para Anclajes SNOMED --
  const [anclajes, setAnclajes] = useState<any[]>([]);
  const [resumenAnclajes, setResumenAnclajes] = useState<any>(null);
  const [estadoAnclajes, setEstadoAnclajes] = useState<string>("pendientes");
  const [selectedAnclaje, setSelectedAnclaje] = useState<any>(null);
  const [linkerTerm, setLinkerTerm] = useState("");
  const [linkerSlot, setLinkerSlot] = useState("");
  const [linkerResults, setLinkerResults] = useState<any[]>([]);

  const fetchEntities = async () => {
    try {
      setStatusMsg("Cargando lista de entidades...");
      const res = await api.getUncuratedEntities();
      setEntities(res.entities || []);
      setStatusMsg("");
    } catch (e: any) {
      setStatusMsg("Error: " + e.message);
    }
  };

  const fetchChunks = async () => {
    try {
      setStatusMsg("Cargando hechos clínicos...");
      const res = await api.getChunks(200, 0); // Cargamos primeros 200 para demo
      setChunks(res.chunks || []);
      setStatusMsg("");
    } catch (e: any) {
      setStatusMsg("Error cargando chunks: " + e.message);
    }
  };

  const fetchAnclajes = async (estado = estadoAnclajes) => {
    try {
      setStatusMsg("Cargando propuestas de anclaje...");
      const res = await api.getAnclajes(estado, 200, 0);
      setAnclajes(res.anclajes || []);
      setResumenAnclajes(res.resumen || null);
      setStatusMsg("");
    } catch (e: any) {
      setStatusMsg("Error cargando anclajes: " + e.message);
    }
  };

  useEffect(() => {
    if (activeTab === "curation") fetchEntities();
    if (activeTab === "chunks") fetchChunks();
    if (activeTab === "anclajes") fetchAnclajes();
  }, [activeTab]);

  // -- Handlers Curación --
  const selectEntity = async (name: string) => {
    try {
      setStatusMsg(`Cargando detalles de ${name}...`);
      setSelectedEntity(null);
      setUmlsPropuestas([]);
      setSelectedSynonyms([]);
      
      const res = await api.getEntityDetails(name);
      setSelectedEntity(res);
      setStatusMsg("");
    } catch (e: any) {
      setStatusMsg("Error: " + e.message);
    }
  };

  const buscarUmls = async () => {
    if (!selectedEntity) return;
    try {
      setStatusMsg("Traduciendo y buscando en UMLS...");
      const res = await api.manualSearchUmls(selectedEntity.name);
      setUmlsPropuestas(res.results || []);
      setStatusMsg("");
    } catch (e: any) {
      setStatusMsg("Error: " + e.message);
    }
  };

  const handleDecision = async (accion: string, cui?: string, nombre_oficial?: string) => {
    if (!selectedEntity) return;
    try {
      setStatusMsg("Guardando decisión...");
      await api.curateEntity(selectedEntity.name, { 
        accion, 
        cui, 
        nombre_oficial, 
        sinonimos_seleccionados: selectedSynonyms 
      });
      setSelectedEntity(null);
      await fetchEntities();
    } catch (e: any) {
      setStatusMsg("Error: " + e.message);
    }
  };

  // -- Handlers Anclajes --
  const selectAnclaje = (a: any) => {
    setSelectedAnclaje(a);
    setLinkerTerm(a.entidad || "");
    setLinkerSlot(a.slot || "");
    setLinkerResults([]);
  };

  const decidirAnclaje = async (accion: string, cand?: any) => {
    if (!selectedAnclaje) return;
    try {
      setStatusMsg("Guardando decisión...");
      await api.decidirAnclaje({
        entidad: selectedAnclaje.entidad,
        accion,
        ...(cand ? { sctid: cand.sctid, cui: cand.cui, termino: cand.termino } : {}),
      });
      setSelectedAnclaje(null);
      await fetchAnclajes();
    } catch (e: any) {
      setStatusMsg("Error: " + e.message);
    }
  };

  const buscarLinker = async () => {
    if (!linkerTerm) return;
    try {
      setStatusMsg("Buscando en terminología local (UMLS 2026AA)...");
      const res = await api.linkerSearch(linkerTerm, linkerSlot);
      setLinkerResults(res.results || []);
      setStatusMsg(res.results?.length ? "" : "Sin candidatos en la terminología local.");
    } catch (e: any) {
      setStatusMsg("Error: " + e.message);
    }
  };

  // -- Handlers Chunks --
  const selectChunk = async (id: string) => {
    try {
      setStatusMsg(`Cargando fragmento ${id}...`);
      setSelectedChunk(null);
      const res = await api.getChunkDetails(id);
      setSelectedChunk(res);
      setStatusMsg("");
    } catch (e: any) {
      setStatusMsg("Error: " + e.message);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "#fff" }}>
      {/* Tabs */}
      <div style={{ display: "flex", background: "#f8f9fa", borderBottom: "1px solid #dee2e6" }}>
        <button 
          onClick={() => setActiveTab("explorer")}
          style={{ padding: "12px 20px", border: "none", background: activeTab === "explorer" ? "#fff" : "transparent", cursor: "pointer", fontWeight: activeTab === "explorer" ? "bold" : "normal", borderRight: "1px solid #dee2e6", borderBottom: activeTab === "explorer" ? "2px solid #0056b3" : "none", color: "#111" }}>
          Visor de Grafo
        </button>
        <button 
          onClick={() => setActiveTab("chunks")}
          style={{ padding: "12px 20px", border: "none", background: activeTab === "chunks" ? "#fff" : "transparent", cursor: "pointer", fontWeight: activeTab === "chunks" ? "bold" : "normal", borderRight: "1px solid #dee2e6", borderBottom: activeTab === "chunks" ? "2px solid #0056b3" : "none", color: "#111" }}>
          Explorador de Hechos
        </button>
        <button
          onClick={() => setActiveTab("curation")}
          style={{ padding: "12px 20px", border: "none", background: activeTab === "curation" ? "#fff" : "transparent", cursor: "pointer", fontWeight: activeTab === "curation" ? "bold" : "normal", borderRight: "1px solid #dee2e6", borderBottom: activeTab === "curation" ? "2px solid #0056b3" : "none", color: "#111" }}>
          Bandeja de Curación
        </button>
        <button
          onClick={() => setActiveTab("anclajes")}
          style={{ padding: "12px 20px", border: "none", background: activeTab === "anclajes" ? "#fff" : "transparent", cursor: "pointer", fontWeight: activeTab === "anclajes" ? "bold" : "normal", borderBottom: activeTab === "anclajes" ? "2px solid #0056b3" : "none", color: "#111" }}>
          Anclajes SNOMED
        </button>
      </div>

      {/* Content */}
      <div style={{ flex: 1, overflow: "hidden" }}>
        {activeTab === "explorer" && <GraphExplorer />}
        
        {activeTab === "curation" && (
          <div style={{ padding: "20px", display: "flex", gap: "20px", height: "100%", overflowY: "auto", background: "#fff" }}>
            
            <div style={{ flex: 1, border: "1px solid #ced4da", padding: "15px", borderRadius: "8px", overflowY: "auto", background: "#f8f9fa" }}>
              <h2 style={{ color: "#111" }}>📋 Entidades Pendientes ({entities.length})</h2>
              {statusMsg && <p style={{ color: "#0056b3", fontWeight: "bold" }}>{statusMsg}</p>}
              
              <ul style={{ listStyleType: "none", padding: 0 }}>
                {entities.map(e => (
                  <li 
                    key={e.name} 
                    onClick={() => selectEntity(e.name)}
                    style={{ 
                      padding: "12px", 
                      borderBottom: "1px solid #dee2e6", 
                      cursor: "pointer",
                      background: selectedEntity?.name === e.name ? "#e7f1ff" : "#fff",
                      display: "flex",
                      justifyContent: "space-between",
                      color: "#111"
                    }}
                  >
                    <strong>{e.name}</strong>
                    <span style={{ fontSize: "0.8em", color: "#111", background: "#e9ecef", padding: "4px 8px", borderRadius: "4px", border: "1px solid #ced4da" }}>
                      {e.tipo_entidad}
                    </span>
                  </li>
                ))}
              </ul>
              {entities.length === 0 && !statusMsg && <p style={{color: "#111"}}>¡Todo al día! No hay entidades pendientes.</p>}
            </div>

            <div style={{ flex: 2, border: "1px solid #ced4da", padding: "15px", borderRadius: "8px", overflowY: "auto", background: "#fff" }}>
              {selectedEntity ? (
                <div>
                  <h2 style={{ color: "#111" }}>Detalles de: <code>{selectedEntity.name}</code></h2>
                  <div style={{ background: "#f8f9fa", padding: "15px", borderRadius: "8px", marginBottom: "20px", border: "1px solid #ced4da", color: "#111" }}>
                    <p><strong>Tipo de Entidad:</strong> {selectedEntity.datos.tipo_entidad}</p>
                    <p><strong>Alias Registrados:</strong> {selectedEntity.datos.alias?.join(", ") || "Ninguno"}</p>
                  </div>
                  
                  {selectedEntity.sinonimos_sugeridos && selectedEntity.sinonimos_sugeridos.length > 0 && (
                    <div style={{ marginBottom: "20px", padding: "15px", background: "#fff3cd", border: "1px solid #ffeeba", borderRadius: "8px", color: "#856404" }}>
                      <h4 style={{ margin: "0 0 10px 0" }}>🧩 Sinónimos detectados en el grafo:</h4>
                      <p style={{ margin: "0 0 10px 0", fontSize: "0.9em" }}>Marca aquellos que signifiquen lo mismo para fusionarlos en esta entidad.</p>
                      {selectedEntity.sinonimos_sugeridos.map((syn: string) => (
                        <label key={syn} style={{ display: "block", marginBottom: "8px", cursor: "pointer", color: "#111" }}>
                          <input 
                            type="checkbox" 
                            checked={selectedSynonyms.includes(syn)}
                            onChange={() => setSelectedSynonyms(prev => prev.includes(syn) ? prev.filter(s => s !== syn) : [...prev, syn])}
                            style={{ marginRight: "8px" }}
                          />
                          {syn}
                        </label>
                      ))}
                    </div>
                  )}

                  <div style={{ borderTop: "1px solid #ced4da", paddingTop: "20px" }}>
                    <h3 style={{ color: "#111" }}>Mapeo UMLS Oficial</h3>
                    <button onClick={buscarUmls} style={{ padding: "10px 16px", background: "#0056b3", color: "white", border: "none", borderRadius: "4px", cursor: "pointer", marginBottom: "15px", fontWeight: "bold" }}>
                      Traducir y Buscar en UMLS
                    </button>
                    
                    {umlsPropuestas.length > 0 && (
                      <ul style={{ listStyleType: "none", padding: 0 }}>
                        {umlsPropuestas.map((p: any, idx: number) => (
                          <li key={idx} style={{ marginBottom: "10px", padding: "15px", border: "1px solid #ced4da", background: "#f8f9fa", borderRadius: "8px", color: "#111" }}>
                            <strong>{p.nombre}</strong> (CUI: {p.cui})
                            <br/>
                            <button onClick={() => handleDecision("aprobar", p.cui, p.nombre)} style={{ marginTop: "10px", background: "#198754", color: "white", border: "none", padding: "8px 16px", borderRadius: "4px", cursor: "pointer", fontWeight: "bold" }}>
                              Aprobar y Fusionar como Golden
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                    
                    <div style={{ marginTop: "20px" }}>
                      <button onClick={() => handleDecision("rechazar")} style={{ background: "#dc3545", color: "white", border: "none", padding: "8px 15px", borderRadius: "4px", cursor: "pointer", fontWeight: "bold" }}>
                        Ignorar Entidad (Rechazar)
                      </button>
                    </div>
                  </div>

                </div>
              ) : (
                <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#6c757d" }}>
                  <p>Selecciona una entidad de la lista para curarla.</p>
                </div>
              )}
            </div>

          </div>
        )}

        {activeTab === "anclajes" && (
          <div style={{ padding: "20px", display: "flex", flexDirection: "column", gap: "12px", height: "100%", overflow: "hidden", background: "#fff" }}>

            <div style={{ padding: "10px 14px", background: "#e7f1ff", border: "1px solid #b6d4fe", borderRadius: "8px", color: "#084298", fontSize: "0.9em" }}>
              Propuestas del pipeline terminológico local (UMLS 2026AA, SCTID como identidad).
              Las decisiones se escriben en <code>anclajes.jsonl</code> y se aplican a la recuperación
              tras reconstruir el grafo (<code>python -m asistente_vih.ingest.construir_grafo --solo-grafo</code>).
            </div>

            <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
              <select value={estadoAnclajes}
                      onChange={(e) => { setEstadoAnclajes(e.target.value); setSelectedAnclaje(null); fetchAnclajes(e.target.value); }}
                      style={{ padding: "8px", border: "1px solid #ced4da", borderRadius: "4px", color: "#111" }}>
                <option value="pendientes">Pendientes de revisión</option>
                <option value="sin_anclar">Sin anclar (resolver a mano)</option>
                <option value="curados">Aprobados</option>
                <option value="rechazados">Rechazados</option>
              </select>
              {resumenAnclajes && (
                <span style={{ fontSize: "0.9em", color: "#444" }}>
                  {resumenAnclajes.pendientes} pendientes · {resumenAnclajes.sin_anclar} sin anclar ·{" "}
                  {resumenAnclajes.curados} aprobados · {resumenAnclajes.rechazados} rechazados · {resumenAnclajes.total} total
                </span>
              )}
              {statusMsg && <span style={{ color: "#0056b3", fontWeight: "bold" }}>{statusMsg}</span>}
            </div>

            <div style={{ display: "flex", gap: "20px", flex: 1, overflow: "hidden" }}>

              <div style={{ flex: 1, border: "1px solid #ced4da", padding: "15px", borderRadius: "8px", overflowY: "auto", background: "#f8f9fa" }}>
                <ul style={{ listStyleType: "none", padding: 0, margin: 0 }}>
                  {anclajes.map((a) => {
                    const badge = METODO_BADGE[a.metodo] || { bg: "#e9ecef", label: a.metodo || "—" };
                    return (
                      <li key={a.entidad} onClick={() => selectAnclaje(a)}
                          style={{ padding: "10px", borderBottom: "1px solid #dee2e6", cursor: "pointer",
                                   background: selectedAnclaje?.entidad === a.entidad ? "#e7f1ff" : "#fff", color: "#111" }}>
                        <div style={{ display: "flex", justifyContent: "space-between", gap: "8px" }}>
                          <strong>{a.entidad}</strong>
                          <span style={{ fontSize: "0.8em", background: badge.bg, padding: "3px 8px", borderRadius: "4px", whiteSpace: "nowrap" }}>
                            {a.sin_anclar ? "sin anclar" : badge.label}
                          </span>
                        </div>
                        {!a.sin_anclar && (
                          <div style={{ fontSize: "0.85em", color: "#555", marginTop: "3px" }}>
                            → {a.termino_umls} · SCTID {a.sctid || "—"} · score {a.score?.toFixed ? a.score.toFixed(2) : a.score}
                          </div>
                        )}
                        <div style={{ fontSize: "0.8em", color: "#888", marginTop: "2px" }}>slot: {a.slot || "—"}</div>
                      </li>
                    );
                  })}
                </ul>
                {anclajes.length === 0 && !statusMsg && <p style={{ color: "#111" }}>Nada en este estado.</p>}
              </div>

              <div style={{ flex: 2, border: "1px solid #ced4da", padding: "15px", borderRadius: "8px", overflowY: "auto", background: "#fff" }}>
                {selectedAnclaje ? (
                  <div style={{ color: "#111" }}>
                    <h2 style={{ marginTop: 0 }}>Anclaje de: <code>{selectedAnclaje.entidad}</code></h2>

                    {!selectedAnclaje.sin_anclar && (
                      <div style={{ background: "#f8f9fa", padding: "15px", borderRadius: "8px", border: "1px solid #ced4da", marginBottom: "15px" }}>
                        <p style={{ margin: "0 0 6px 0" }}><strong>Propuesta:</strong> {selectedAnclaje.termino_umls}</p>
                        <p style={{ margin: "0 0 6px 0" }}><strong>SCTID:</strong> <code>{selectedAnclaje.sctid || "—"}</code> · <strong>CUI:</strong> <code>{selectedAnclaje.cui || "—"}</code></p>
                        <p style={{ margin: "0 0 6px 0" }}><strong>Método:</strong> {selectedAnclaje.metodo} · <strong>Score:</strong> {selectedAnclaje.score} · <strong>Fuente:</strong> {selectedAnclaje.sab} · <strong>Release:</strong> {selectedAnclaje.release}</p>
                        {selectedAnclaje.motivo_juez && <p style={{ margin: 0 }}><strong>Motivo del juez:</strong> {selectedAnclaje.motivo_juez}</p>}
                      </div>
                    )}

                    <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
                      {!selectedAnclaje.sin_anclar && (
                        <button onClick={() => decidirAnclaje("aprobar")}
                                style={{ background: "#198754", color: "white", border: "none", padding: "10px 18px", borderRadius: "4px", cursor: "pointer", fontWeight: "bold" }}>
                          ✓ Aprobar propuesta
                        </button>
                      )}
                      <button onClick={() => decidirAnclaje("rechazar")}
                              style={{ background: "#dc3545", color: "white", border: "none", padding: "10px 18px", borderRadius: "4px", cursor: "pointer", fontWeight: "bold" }}>
                        ✕ Rechazar
                      </button>
                    </div>

                    <div style={{ borderTop: "1px solid #ced4da", paddingTop: "15px" }}>
                      <h3 style={{ marginTop: 0 }}>Buscar en terminología local</h3>
                      <div style={{ display: "flex", gap: "8px", marginBottom: "12px" }}>
                        <input value={linkerTerm} onChange={(e) => setLinkerTerm(e.target.value)}
                               onKeyDown={(e) => e.key === "Enter" && buscarLinker()}
                               style={{ flex: 1, padding: "8px", border: "1px solid #ced4da", borderRadius: "4px", color: "#111" }} />
                        <select value={linkerSlot} onChange={(e) => setLinkerSlot(e.target.value)}
                                style={{ padding: "8px", border: "1px solid #ced4da", borderRadius: "4px", color: "#111" }}>
                          <option value="">(sin slot)</option>
                          <option value="farmaco">farmaco</option>
                          <option value="condicion">condicion</option>
                          <option value="intervencion">intervencion</option>
                          <option value="poblacion">poblacion</option>
                          <option value="resultado">resultado</option>
                        </select>
                        <button onClick={buscarLinker}
                                style={{ padding: "8px 16px", background: "#0056b3", color: "white", border: "none", borderRadius: "4px", cursor: "pointer", fontWeight: "bold" }}>
                          Buscar
                        </button>
                      </div>
                      {linkerResults.map((c: any, i: number) => (
                        <div key={i} style={{ padding: "12px", border: "1px solid #ced4da", borderRadius: "8px", background: "#f8f9fa", marginBottom: "8px" }}>
                          <strong>{c.termino}</strong> — SCTID <code>{c.sctid || "—"}</code> · CUI <code>{c.cui}</code>
                          <div style={{ fontSize: "0.85em", color: "#555", margin: "4px 0" }}>
                            score {c.score} · {c.metodo} · {c.sab} · grupos: {c.grupos?.join(", ") || "—"}
                          </div>
                          <button onClick={() => decidirAnclaje("aprobar", c)}
                                  style={{ background: "#198754", color: "white", border: "none", padding: "6px 14px", borderRadius: "4px", cursor: "pointer", fontWeight: "bold" }}>
                            Anclar con este concepto
                          </button>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#6c757d" }}>
                    <p>Selecciona una propuesta. Prioridad: método "juez" y scores bajos primero.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === "chunks" && (
          <div style={{ padding: "20px", display: "flex", gap: "20px", height: "100%", overflowY: "auto", background: "#fff" }}>
            
            {/* Lista de Chunks */}
            <div style={{ flex: 1, border: "1px solid #ced4da", padding: "15px", borderRadius: "8px", overflowY: "auto", background: "#f8f9fa" }}>
              <h2 style={{ color: "#111" }}>📑 Hechos Clínicos</h2>
              {statusMsg && <p style={{ color: "#0056b3", fontWeight: "bold" }}>{statusMsg}</p>}
              
              <ul style={{ listStyleType: "none", padding: 0 }}>
                {chunks.map(c => (
                  <li 
                    key={c.id} 
                    onClick={() => selectChunk(c.id)}
                    style={{ 
                      padding: "12px", 
                      borderBottom: "1px solid #dee2e6", 
                      cursor: "pointer",
                      background: selectedChunk?.id === c.id ? "#e7f1ff" : "#fff",
                      color: "#111"
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "5px" }}>
                      <strong>{c.id}</strong>
                      <span style={{ fontSize: "0.8em", color: "#666" }}>{c.guia}</span>
                    </div>
                    <p style={{ margin: 0, fontSize: "0.9em", color: "#444" }}>{c.texto_preview}</p>
                  </li>
                ))}
              </ul>
            </div>

            {/* Detalles del Chunk */}
            <div style={{ flex: 2, border: "1px solid #ced4da", padding: "15px", borderRadius: "8px", overflowY: "auto", background: "#fff" }}>
              {selectedChunk ? (
                <div>
                  <h2 style={{ color: "#111", borderBottom: "2px solid #0056b3", paddingBottom: "10px", marginBottom: "10px" }}>Detalles del {selectedChunk.id}</h2>
                  
                  <div style={{ display: "flex", gap: "15px", marginBottom: "15px", fontSize: "0.9em", color: "#444" }}>
                    <span style={{ background: "#e9ecef", padding: "4px 8px", borderRadius: "4px" }}><strong>Guía:</strong> {selectedChunk.metadata?.guia || "Desconocida"}</span>
                    <span style={{ background: "#e9ecef", padding: "4px 8px", borderRadius: "4px" }}><strong>Sección:</strong> {selectedChunk.metadata?.seccion || "N/A"}</span>
                    <span style={{ background: "#e9ecef", padding: "4px 8px", borderRadius: "4px" }}><strong>Página:</strong> {selectedChunk.metadata?.pagina || "N/A"}</span>
                  </div>
                  
                  <div style={{ background: "#f8f9fa", padding: "15px", borderRadius: "8px", marginBottom: "20px", border: "1px solid #ced4da" }}>
                    <h3 style={{ marginTop: 0, color: "#111" }}>Texto Original</h3>
                    <p style={{ whiteSpace: "pre-wrap", color: "#333", lineHeight: "1.6" }}>{selectedChunk.texto}</p>
                  </div>

                  <h3 style={{ color: "#111" }}>Aserciones Extraídas (Reificación)</h3>
                  {selectedChunk.relaciones && selectedChunk.relaciones.length > 0 ? (
                    <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "10px", color: "#111", fontSize: "0.9em" }}>
                      <thead>
                        <tr style={{ background: "#e9ecef", borderBottom: "2px solid #dee2e6" }}>
                          <th style={{ padding: "10px", textAlign: "left", border: "1px solid #ced4da" }}>Tipo</th>
                          <th style={{ padding: "10px", textAlign: "left", border: "1px solid #ced4da" }}>Intervención</th>
                          <th style={{ padding: "10px", textAlign: "left", border: "1px solid #ced4da" }}>Relación Base</th>
                          <th style={{ padding: "10px", textAlign: "left", border: "1px solid #ced4da" }}>Resultado Esperado</th>
                          <th style={{ padding: "10px", textAlign: "left", border: "1px solid #ced4da" }}>Población Diana</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selectedChunk.relaciones.map((r: any, idx: number) => (
                          <tr key={idx} style={{ borderBottom: "1px solid #dee2e6" }}>
                            <td style={{ padding: "10px", border: "1px solid #ced4da" }}>
                                <span style={{ background: "#d1e7dd", padding: "4px 8px", borderRadius: "4px", fontSize: "0.85em" }}>{r.tipo_asercion}</span>
                            </td>
                            <td style={{ padding: "10px", border: "1px solid #ced4da" }}><strong>{r.intervencion}</strong></td>
                            <td style={{ padding: "10px", border: "1px solid #ced4da", color: "#0056b3", fontWeight: "bold" }}>{r.relacion_base}</td>
                            <td style={{ padding: "10px", border: "1px solid #ced4da" }}><strong>{r.resultado_esperado}</strong></td>
                            <td style={{ padding: "10px", border: "1px solid #ced4da", fontSize: "0.9em" }}>{r.poblacion_diana}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <p style={{ color: "#666" }}>El modelo no extrajo ninguna aserción clínica de este fragmento.</p>
                  )}
                  
                </div>
              ) : (
                <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", color: "#6c757d" }}>
                  <p>Selecciona un fragmento de la lista para ver sus hechos clínicos.</p>
                </div>
              )}
            </div>

          </div>
        )}
      </div>
    </div>
  );
}
