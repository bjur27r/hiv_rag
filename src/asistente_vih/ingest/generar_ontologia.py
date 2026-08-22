"""Script para generar el diccionario ontológico base usando DeepSeek.

Muestrea el 10% de los chunks generados, los envía a DeepSeek para 
extraer posibles tipos de entidades y relaciones, y los consolida en un diccionario.

Uso:
    export DEEPSEEK_API_KEY="tu_clave_aqui"
    python -m asistente_vih.ingest.generar_ontologia
"""
import json
import os
import random
from collections import Counter
from typing import Any

from openai import OpenAI
from ..config import CHUNKS_PATH, ARTIFACTS_DIR

ONTOLOGY_OUT_PATH = ARTIFACTS_DIR / "ontology.json"

PROMPT_SISTEMA = """Eres un experto en extracción de ontologías médicas para VIH.
Tu objetivo es leer un fragmento de guía clínica y determinar qué TIPOS de entidades y TIPOS de relaciones están presentes, para alimentar un Grafo de Conocimiento Médico (HippoRAG/CatRAG).

NO inventes nombres específicos de pacientes, extrae los conceptos generales.
NO respondas con texto, SOLAMENTE devuelve un objeto JSON estrictamente formateado así:
{
  "entidades": [
    {"tipo": "Farmaco", "nombre_ejemplo": "Tenofovir"},
    {"tipo": "Biomarcador", "nombre_ejemplo": "CD4"}
  ],
  "relaciones": [
    {"tipo": "CONTRAINDICADO_EN", "origen": "Farmaco", "destino": "Condicion"},
    {"tipo": "AUMENTA_RIESGO_DE", "origen": "Condicion", "destino": "Condicion"}
  ]
}
"""

def procesar_chunk_deepseek(client: OpenAI, texto: str) -> dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": PROMPT_SISTEMA},
                {"role": "user", "content": f"Fragmento:\n{texto}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error procesando chunk: {e}")
        return {"entidades": [], "relaciones": []}

def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: Define DEEPSEEK_API_KEY en el entorno para continuar.")
        return

    print("Iniciando cliente DeepSeek...")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    print(f"Cargando chunks desde {CHUNKS_PATH}...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    
    total = len(chunks)
    muestra = max(1, int(total * 0.10))
    print(f"Total de chunks: {total}. Seleccionando muestra del 10%: {muestra} chunks.")
    
    random.seed(42)  # Para reproducibilidad
    chunks_muestra = random.sample(chunks, k=muestra)
    
    tipos_entidades: Counter = Counter()
    nombres_entidades: set = set()
    tipos_relaciones: Counter = Counter()
    ejemplos_relaciones: set = set()
    
    print("Enviando chunks a DeepSeek (esto tomará un tiempo)...")
    for i, c in enumerate(chunks_muestra, 1):
        texto = c.get("texto", "")
        if not texto: continue
        
        print(f"\rProcesando {i}/{muestra}...", end="", flush=True)
        resultado = procesar_chunk_deepseek(client, texto)
        
        for ent in resultado.get("entidades", []):
            tipos_entidades[ent.get("tipo", "Desconocido")] += 1
            nombres_entidades.add((ent.get("tipo"), ent.get("nombre_ejemplo")))
            
        for rel in resultado.get("relaciones", []):
            t_rel = rel.get("tipo", "REL")
            tipos_relaciones[t_rel] += 1
            ejemplos_relaciones.add(f"{rel.get('origen')} -[{t_rel}]-> {rel.get('destino')}")

    print("\n\nExtracción completada. Consolidando ontología...")
    
    # Quedarnos con los tipos y relaciones más frecuentes para evitar ruido (ej. min 3 menciones)
    tipos_ent_finales = [t for t, count in tipos_entidades.most_common() if count >= 2]
    tipos_rel_finales = [t for t, count in tipos_relaciones.most_common() if count >= 2]
    
    ontologia = {
        "entidades_permitidas": tipos_ent_finales,
        "relaciones_permitidas": tipos_rel_finales,
        "ejemplos_descubiertos": {
            "entidades": list(nombres_entidades)[:50],  # Mostrar max 50 ejemplos
            "relaciones": list(ejemplos_relaciones)[:50]
        }
    }
    
    with open(ONTOLOGY_OUT_PATH, "w", encoding="utf-8") as out:
        json.dump(ontologia, out, ensure_ascii=False, indent=2)
        
    print(f"\n¡Ontología guardada en {ONTOLOGY_OUT_PATH}!")
    print(f"Descubiertos {len(tipos_ent_finales)} tipos de entidades y {len(tipos_rel_finales)} tipos de relaciones frecuentes.")
    print("Por favor, revisa el archivo JSON para ajustar los nombres manualmente.")

if __name__ == "__main__":
    main()
