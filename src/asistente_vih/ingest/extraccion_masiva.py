"""Script para extracción masiva (2 pasos + Reificación).

Lee todos los fragmentos (chunks.jsonl), usa DeepSeek,
y extrae Relaciones n-dimensionales (Aserciones) y un Diccionario de Entidades.
Utiliza asyncio para manejar rate limits.

Uso:
    export DEEPSEEK_API_KEY="tu_clave_aqui"
    python3 -m asistente_vih.ingest.extraccion_masiva
"""

import json
import os
import asyncio
from typing import Any

from openai import AsyncOpenAI
from ..config import CHUNKS_PATH, ARTIFACTS_DIR

RELACIONES_OUT_PATH = ARTIFACTS_DIR / "triplets.jsonl"
DICCIONARIO_OUT_PATH = ARTIFACTS_DIR / "entity_dictionary.json"

MAX_CONCURRENT_REQUESTS = 10 

async def procesar_chunk(
    client: AsyncOpenAI, 
    sem: asyncio.Semaphore, 
    texto_actual: str, 
    texto_previo: str, 
    chunk_id: str
) -> dict:
    async with sem:
        if not texto_actual.strip():
            return {"entidades": [], "aserciones": [], "chunk_id": chunk_id}

        # --- PASO 1: NER ---
        prompt_ner = """Eres un experto en Reconocimiento de Entidades Nombradas (NER) Médico.
Extrae únicamente los conceptos médicos del texto y clasifícalos.
Clases permitidas: Farmaco, Enfermedad, CondicionClinica, Poblacion, Intervencion, ResultadoClinico.
Responde estrictamente en JSON con la clave 'entidades' que sea una lista de objetos {nombre_canonico, tipo}."""

        texto_completo = f"--- CONTEXTO ANTERIOR ---\n{texto_previo}\n\n--- TEXTO ACTUAL ---\n{texto_actual}"

        entidades_json = {"entidades": []}
        try:
            resp_ner = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt_ner},
                    {"role": "user", "content": texto_completo}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            entidades_json = json.loads(resp_ner.choices[0].message.content)
        except Exception as e:
            print(f"Error NER en chunk {chunk_id}: {e}")
            return {"entidades": [], "aserciones": [], "chunk_id": chunk_id, "error": str(e)}

        # --- PASO 2: REIFICACIÓN ---
        prompt_reification = f"""Eres un ingeniero de conocimiento clínico.
Tu tarea es construir ASERCIONES CLÍNICAS (Nodos de Recomendación/Evidencia) a partir del texto y las entidades previas.

Lista de Entidades Detectadas:
{json.dumps(entidades_json.get('entidades', []), ensure_ascii=False)}

Ontología Semi-Cerrada (Relaciones Base Permitidas):
TREATS, CAUSES, PREVENTS, CONTRAINDICATES, INCREASES_RISK, REDUCES_RISK, IS_NON_INFERIOR_TO, IMPROVES, WORSENS, HAS_OUTCOME, IS_DIAGNOSED_BY, HAS_SYMPTOM, OTHER_RELATION.

FORMATO DE SALIDA (ESTRICTAMENTE JSON):
{{
  "aserciones": [
    {{
      "id_asercion": "{chunk_id}_X",
      "tipo_asercion": "RecomendacionTerapeutica | EvidenciaClinica | Advertencia | Diagnostico",
      "evidence_level": "Extrae nivel si lo hay, si no null",
      "is_negated": false,
      "intervencion": "El fármaco o procedimiento principal",
      "poblacion_diana": "Condición de los pacientes a los que aplica",
      "resultado_esperado": "El resultado de aplicar la intervención",
      "relacion_base": "UNA_DE_LAS_PERMITIDAS",
      "descripcion_relacion": "Breve descripción de la aserción"
    }}
  ]
}}
"""
        aserciones_json = {"aserciones": []}
        try:
            resp_rel = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": prompt_reification},
                    {"role": "user", "content": texto_completo}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            aserciones_json = json.loads(resp_rel.choices[0].message.content)
        except Exception as e:
            print(f"Error Reificacion en chunk {chunk_id}: {e}")
            pass

        return {
            "chunk_id": chunk_id,
            "entidades": entidades_json.get("entidades", []),
            "aserciones": aserciones_json.get("aserciones", [])
        }


async def main_async():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY no definida.")
        return

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
        
    print(f"Iniciando extracción masiva (2 Pasos) sobre {len(chunks)} fragmentos...")
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    tareas = []
    for i, c in enumerate(chunks):
        texto_actual = c.get("texto", "")
        texto_previo = chunks[i-1].get("texto", "") if i > 0 else "Inicio del documento."
        chunk_id = c.get("chunk_id", f"chunk_{i}")
        
        tareas.append(procesar_chunk(client, sem, texto_actual, texto_previo, chunk_id))
        
    resultados = await asyncio.gather(*tareas)
    
    # ---------------------------------------------------------
    # CONSOLIDACIÓN Y GUARDADO
    # ---------------------------------------------------------
    diccionario_entidades = {}
    
    print("\nExtracción finalizada. Consolidando base de datos...")
    
    with open(RELACIONES_OUT_PATH, "w", encoding="utf-8") as f_rel:
        for res in resultados:
            if res.get("aserciones"):
                f_rel.write(json.dumps({
                    "chunk_id": res["chunk_id"],
                    "aserciones": res["aserciones"]
                }, ensure_ascii=False) + "\n")
                
            # Consolidar el Diccionario de Entidades
            for ent in res.get("entidades", []):
                canon = ent.get("nombre_canonico") or ent.get("nombre")
                if not canon: continue
                
                canon_key = canon.strip().title() 
                
                if canon_key not in diccionario_entidades:
                    diccionario_entidades[canon_key] = {
                        "nombre_canonico": canon_key,
                        "tipo_entidad": ent.get("tipo") or ent.get("clase") or "Desconocido",
                        "curado": False,
                        "alias": set(),
                        "codigo_umls": "",
                        "relaciones_umls": []
                    }
                
                # Alias
                alias = ent.get("alias_en_texto", [])
                if isinstance(alias, list):
                    for a in alias:
                        if a.strip().title() != canon_key:
                            diccionario_entidades[canon_key]["alias"].add(a.strip())

    for k in diccionario_entidades:
        diccionario_entidades[k]["alias"] = list(diccionario_entidades[k]["alias"])

    with open(DICCIONARIO_OUT_PATH, "w", encoding="utf-8") as f_dicc:
        json.dump(diccionario_entidades, f_dicc, ensure_ascii=False, indent=2)
        
    print(f"¡Éxito! \n- Aserciones guardadas en: {RELACIONES_OUT_PATH}")
    print(f"- Diccionario de Entidades guardado en: {DICCIONARIO_OUT_PATH}")
    print(f"Total entidades únicas consolidadas: {len(diccionario_entidades)}")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
