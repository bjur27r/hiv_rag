import os
import uuid
from asistente_vih.agent.graph import construir_grafo

preguntas = [
    "¿Cuál es la dosis recomendada de Abacavir?",
    "Tengo un paciente con tuberculosis activa en tratamiento con rifampicina. ¿Puedo pautarle Dolutegravir?",
    "¿Qué regímenes antirretrovirales se deben evitar en el embarazo por riesgo de defectos del tubo neural?",
    "Mi paciente va a iniciar tratamiento con tenofovir, pero su FG es de 40. ¿Qué hago?",
    "¿Existen interacciones o contraindicaciones entre Biktarvy y antiácidos?"
]

print("Iniciando Grafo LangGraph con motor Adaptive RAG (Vectorial + CatRAG)...")
app = construir_grafo()

for i, p in enumerate(preguntas, 1):
    print(f"\n\n{'='*80}\nTEST {i}: {p}\n{'='*80}")
    
    # Thread único por pregunta para no mezclar historial
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    estado = {"pregunta": p}
    
    try:
        # Correr el grafo paso a paso
        for event in app.stream(estado, config=config):
            for node_name, node_state in event.items():
                print(f" -> Nodo ejecutado: {node_name}")
            
            # Si el nodo es hitl, el generador devuelve algo en __interrupt__ o simplemente frena.
            # En LangGraph moderno, la pausa se ve al final del loop.
            pass
            
        # Obtener el estado final
        state = app.get_state(config)
        vals = state.values
        
        # Verificar si está pausado
        if len(state.next) > 0 and state.next[0] == "sintetizar": 
            # Si el proximo nodo pendiente es sintetizar, y pasamos por HITL...
            print(f">>> GRAFO PAUSADO POR HITL. Esperando input humano.")
            # Continuamos al siguiente test
            continue
            
        estrategia = vals.get("clasificacion", {}).get("estrategia_recuperacion", "desconocida")
        decision_seria = vals.get("clasificacion", {}).get("decision_seria", False)
        
        print(f"\n* Estrategia de Router: {estrategia.upper()}")
        print(f"* Decisión Seria: {decision_seria}")
        
        veredicto = vals.get("veredicto", {})
        if veredicto:
            print(f"* Verificado Fiel: {veredicto.get('fiel', False)}")
            
        print(f"\n[ RESPUESTA FINAL ]\n{vals.get('respuesta', 'No hay respuesta.')}")
        
    except Exception as e:
        print(f"Error en Test {i}: {e}")
