import { useState, useEffect, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { api } from "../api";

export default function GraphExplorer() {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [searchTerm, setSearchTerm] = useState("");
  const [autocompleteResults, setAutocompleteResults] = useState<string[]>([]);
  const [hops, setHops] = useState<number>(2);
  const [status, setStatus] = useState("Cargando grafo (Top 100 hubs)...");
  
  const [selectedNode, setSelectedNode] = useState<any>(null);
  
  const fgRef = useRef<any>();
  const searchTimeout = useRef<any>(null);

  const loadGraph = async (query = "", targetHops = hops) => {
    try {
      setStatus(query ? `Buscando vecindario de "${query}" a ${targetHops} saltos...` : "Cargando grafo (Top 100 hubs)...");
      const data = await api.getGraphData(query, targetHops);
      setGraphData(data);
      setStatus(data.nodes.length > 0 ? "" : "No se encontraron nodos.");
      setAutocompleteResults([]);
      setSelectedNode(null);
      
      if (data.nodes.length > 0 && fgRef.current) {
        setTimeout(() => fgRef.current.zoomToFit(400, 50), 500);
      }
    } catch (e: any) {
      setStatus("Error cargando grafo: " + e.message);
    }
  };

  useEffect(() => {
    loadGraph("", 2);
  }, []);

  const handleSearchChange = (val: string) => {
    setSearchTerm(val);
    if (searchTimeout.current) clearTimeout(searchTimeout.current);
    
    if (val.length >= 2) {
      searchTimeout.current = setTimeout(async () => {
        try {
          const res = await api.searchGraphNodes(val);
          setAutocompleteResults(res.results || []);
        } catch (e) {
          console.error(e);
        }
      }, 300);
    } else {
      setAutocompleteResults([]);
    }
  };

  const handleNodeClick = (node: any) => {
    setSelectedNode(node);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", width: "100%", position: "relative" }}>
      
      {/* Barra superior de herramientas (Alto Contraste) */}
      <div style={{ padding: "15px", background: "#f8f9fa", borderBottom: "1px solid #ced4da", display: "flex", gap: "15px", alignItems: "center", color: "#111" }}>
        <strong style={{ fontSize: "1.1em" }}>Visor de Ontologías Golden</strong>
        
        <div style={{ position: "relative" }}>
          <input
            type="text"
            placeholder="Buscar entidad, concepto o chunk (Ej: Tenofovir, sct:19030005, TAR_ADULTOS_2022::p6::18)"
            value={searchTerm}
            onChange={(e) => handleSearchChange(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && loadGraph(searchTerm, hops)}
            style={{ padding: "10px", width: "350px", borderRadius: "4px", border: "1px solid #ced4da", color: "#111" }}
          />
          {autocompleteResults.length > 0 && (
            <ul style={{ position: "absolute", top: "100%", left: 0, width: "100%", background: "white", border: "1px solid #ced4da", zIndex: 10, listStyle: "none", padding: 0, margin: 0, maxHeight: "250px", overflowY: "auto", boxShadow: "0 4px 12px rgba(0,0,0,0.15)" }}>
              {autocompleteResults.map(res => (
                <li 
                  key={res} 
                  onClick={() => { setSearchTerm(res); loadGraph(res, hops); }}
                  style={{ padding: "12px", cursor: "pointer", borderBottom: "1px solid #e9ecef", color: "#111", fontWeight: "500" }}
                  onMouseEnter={(e) => (e.currentTarget.style.background = "#e7f1ff")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "white")}
                >
                  {res}
                </li>
              ))}
            </ul>
          )}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
          <label style={{ fontWeight: "bold" }}>Saltos (Hops):</label>
          <input 
            type="number" 
            min={1} 
            max={5} 
            value={hops} 
            onChange={(e) => setHops(parseInt(e.target.value) || 1)}
            style={{ width: "60px", padding: "8px", border: "1px solid #ced4da", borderRadius: "4px", color: "#111" }}
          />
        </div>

        <button onClick={() => loadGraph(searchTerm, hops)} style={{ padding: "10px 15px", background: "#0056b3", color: "white", border: "none", borderRadius: "4px", cursor: "pointer", fontWeight: "bold" }}>
          Explorar Vecindario
        </button>
        
        <button onClick={() => { setSearchTerm(""); loadGraph("", hops); }} style={{ padding: "10px 15px", background: "#495057", color: "white", border: "none", borderRadius: "4px", cursor: "pointer", fontWeight: "bold" }}>
          Ver Hubs Globales
        </button>
        
        <span style={{ color: "#0056b3", marginLeft: "10px", fontWeight: "bold" }}>{status}</span>
      </div>
      
      {/* Contenedor principal */}
      <div style={{ flex: 1, backgroundColor: "#1e1e1e", overflow: "hidden", position: "relative" }}>
        
        {/* Grafo */}
        {graphData.nodes.length > 0 && (
          <ForceGraph2D
            ref={fgRef}
            graphData={graphData}
            nodeLabel={(n: any) => `${n.label || n.id} (${n.type})`}
            nodeAutoColorBy="type"
            linkColor={() => "rgba(255,255,255,0.25)"}
            linkDirectionalArrowLength={4}
            linkDirectionalArrowRelPos={1}
            linkLabel={(link: any) => link.label}
            onNodeClick={handleNodeClick}
            nodeVal={(n: any) => {
              // Si es el nodo seleccionado, lo hacemos mucho más grande
              if (selectedNode && n.id === selectedNode.id) return 30;
              if (n.type === "Chunk") return 15;
              if (n.type === "Recomendacion") return 10;
              if (n.type === "Concepto") return 8;   // capa SNOMED
              return 5;
            }}
          />
        )}

        {/* Panel lateral derecho (Información del Nodo) - Alto Contraste */}
        {selectedNode && (
          <div style={{ position: "absolute", top: "20px", right: "20px", width: "350px", background: "#ffffff", borderRadius: "8px", boxShadow: "0 6px 20px rgba(0,0,0,0.4)", padding: "20px", zIndex: 5, border: "1px solid #ced4da" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "15px" }}>
              <h3 style={{ margin: 0, color: "#111", wordBreak: "break-word", fontSize: "1.3em" }}>
                {selectedNode.label || selectedNode.id}
                {selectedNode.label && selectedNode.label !== selectedNode.id && (
                  <div style={{ fontSize: "0.6em", color: "#6c757d", fontWeight: "normal" }}>{selectedNode.id}</div>
                )}
              </h3>
              <button 
                onClick={() => setSelectedNode(null)} 
                style={{ background: "transparent", border: "none", fontSize: "1.5em", cursor: "pointer", color: "#6c757d", fontWeight: "bold" }}
              >
                ✕
              </button>
            </div>
            
            <div style={{ display: "flex", flexDirection: "column", gap: "12px", fontSize: "1em", color: "#111" }}>
              <p style={{ margin: 0 }}>
                <strong>Tipo:</strong> 
                <span style={{ marginLeft: "8px", background: "#e9ecef", padding: "4px 10px", borderRadius: "12px", fontSize: "0.9em", border: "1px solid #ced4da" }}>
                  {selectedNode.type}
                </span>
              </p>
              
              {selectedNode.type !== "Chunk" && (
                <p style={{ margin: 0 }}>
                  <strong>Estado:</strong> 
                  {selectedNode.curado ? (
                    <span style={{ marginLeft: "8px", color: "#198754", fontWeight: "bold" }}>✓ Curado (Golden)</span>
                  ) : (
                    <span style={{ marginLeft: "8px", color: "#d39e00", fontWeight: "bold" }}>⚠ No Curado</span>
                  )}
                </p>
              )}

              {selectedNode.sctid && (
                <p style={{ margin: 0 }}>
                  <strong>SNOMED CT:</strong> <code>{selectedNode.sctid}</code>
                  {selectedNode.anclaje_metodo && (
                    <span style={{ marginLeft: "8px", fontSize: "0.85em", background: "#e7f1ff", padding: "2px 8px", borderRadius: "10px", border: "1px solid #b6d4fe" }}>
                      anclaje: {selectedNode.anclaje_metodo}
                    </span>
                  )}
                </p>
              )}

              {selectedNode.cui && (
                <p style={{ margin: 0 }}>
                  <strong>UMLS CUI:</strong> <code>{selectedNode.cui}</code>
                </p>
              )}

              {selectedNode.attrs && Object.keys(selectedNode.attrs).filter(k => !["nombre", "sctid"].includes(k)).length > 0 && (
                <div style={{ marginTop: "6px", padding: "10px", background: "#f8f9fa", borderRadius: "8px", border: "1px solid #ced4da" }}>
                  <strong>Atributos del grafo:</strong>
                  <table style={{ width: "100%", marginTop: "6px", fontSize: "0.9em", borderCollapse: "collapse" }}>
                    <tbody>
                      {Object.entries(selectedNode.attrs)
                        .filter(([k]) => !["nombre", "sctid"].includes(k))
                        .map(([k, v]) => (
                          <tr key={k}>
                            <td style={{ padding: "3px 8px 3px 0", color: "#6c757d", verticalAlign: "top", whiteSpace: "nowrap" }}>{k}</td>
                            <td style={{ padding: "3px 0", color: "#111", wordBreak: "break-word" }}>{String(v)}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              )}

              {selectedNode.alias && selectedNode.alias.length > 0 && (
                <div style={{ marginTop: "10px", padding: "10px", background: "#f8f9fa", borderRadius: "8px", border: "1px solid #ced4da" }}>
                  <strong>Alias Registrados:</strong>
                  <ul style={{ margin: "8px 0 0 0", paddingLeft: "20px", color: "#333" }}>
                    {selectedNode.alias.map((al: string, i: number) => (
                      <li key={i}>{al}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
