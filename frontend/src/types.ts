export interface Usuario { id: number; email: string; }
export interface Thread { id: string; title: string; }
export interface Cita {
  n: number; chunk_id: string; guia: string; seccion: string; pagina: number; texto: string;
  vigencia?: string;
  reasoning_trace?: string[];
}

export interface Anclaje {
  entidad: string; slot?: string; curado?: boolean; rechazado?: boolean;
  sin_anclar?: boolean; cui?: string; sctid?: string; termino_umls?: string;
  score?: number; metodo?: string; sab?: string; motivo_juez?: string; release?: string;
}

export interface CandidatoLinker {
  cui: string; sctid: string; termino: string; score: number;
  sab: string; metodo: string; grupos: string[];
}
export interface DetalleProceso {
  clasificacion?: any;   // salida del router (nivel, estrategia, multi_aspecto...)
  suficiencia?: any;     // veredicto CRAG (accion, motivo)
  veredicto?: any;       // veredicto completo del verificador (+lentes)
  ecl?: any;             // resumen del analisis de clase SNOMED, si aplico
}

export interface Mensaje {
  id?: number;
  role: "medico" | "asistente";
  content: string;
  citations?: Cita[] | null;
  model?: string | null;
  veredicto?: string | null;
  abstenida?: boolean;
  pasos?: string[];        // traza agentica (solo en streaming en vivo)
  ruta?: string[];         // ruta completa del grafo (persistente en la sesion)
  detalle?: DetalleProceso;
  streaming?: boolean;
  hitl?: any;              // payload de pausa HITL
}
export interface ModeloItem { id: string; label: string; }
export interface Proveedor {
  provider: string; label: string; disponible: boolean; modelos: ModeloItem[];
}
export type SSEvento =
  | { type: "paso"; nodo: string }
  | { type: "respuesta"; texto: string; abstenida: boolean; veredicto: string | null; citas: Cita[]; ruta: string[]; detalle?: DetalleProceso }
  | { type: "hitl"; payload: any }
  | { type: "error"; mensaje: string }
  | { type: "fin" };
