"""Cliente REST para la API de UMLS (National Library of Medicine).

Permite buscar conceptos médicos, validar terminología y obtener 
los códigos CUI (Concept Unique Identifier) para curar el Grafo Golden.
"""

import os
import requests
from typing import List, Dict, Any

class UMLSClient:
    BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("UMLS_API_KEY")
        if not self.api_key:
            raise ValueError("No se encontró UMLS_API_KEY en el entorno.")
            
    def search_term(self, term: str, exact_match: bool = False) -> List[Dict[str, Any]]:
        """
        Busca un término médico en UMLS, traduciéndolo al inglés primero para mejor match.
        """
        # 1. Traducción automática usando OpenAI
        term_en = term
        try:
            from openai import OpenAI
            client = OpenAI()
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"Traduce el siguiente término médico del español al inglés de la forma más precisa y oficial posible (solo devuelve la traducción, sin explicaciones ni comillas): {term}"}],
                temperature=0.0
            )
            term_en = res.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error en traducción de UMLS: {e}")
            
        endpoint = f"{self.BASE_URL}/search/current"
        
        search_type = "exact" if exact_match else "words"
        params = {
            "string": term_en,
            "apiKey": self.api_key,
            "searchType": search_type,
            "sabs": "SNOMEDCT_US,RXNORM,MSH,MDRSPA,SCTSPA", # Agregado vocabularios en español
            "returnIdType": "concept"
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("result", {}).get("results", [])
            
            # Formatear la salida para que sea fácil de consumir en el front-end
            conceptos = []
            for r in results:
                conceptos.append({
                    "cui": r.get("ui"),
                    "nombre": r.get("name"),
                    "uri": r.get("uri")
                })
                
            return conceptos
            
        except requests.exceptions.RequestException as e:
            print(f"Error conectando a UMLS: {e}")
            return []

    def get_concept_details(self, cui: str) -> Dict[str, Any]:
        """Obtiene detalles específicos de un CUI."""
        endpoint = f"{self.BASE_URL}/content/current/CUI/{cui}"
        params = {"apiKey": self.api_key}
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            r = data.get("result", {})
            return {
                "cui": r.get("ui"),
                "nombre": r.get("name"),
                "definicion": r.get("definitions", "No disponible")
            }
        except Exception as e:
            print(f"Error obteniendo detalles del CUI {cui}: {e}")
            return {}

    def get_concept_relations(self, cui: str) -> List[Dict[str, Any]]:
        """Obtiene las relaciones semánticas (isa, has_ingredient, etc.) de un CUI desde UMLS."""
        endpoint = f"{self.BASE_URL}/content/current/CUI/{cui}/relations"
        params = {"apiKey": self.api_key}
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            results = data.get("result", [])
            relaciones = []
            
            for r in results:
                # Filtrar para no saturar: quedarse solo con relaciones fuertes como RO (has relation), PAR (has parent), CHD (has child)
                rel_label = r.get("additionalRelationLabel", r.get("relationLabel", ""))
                related_cui = r.get("relatedId", "").split("/")[-1]
                related_name = r.get("relatedIdName", "")
                
                if rel_label and related_cui and related_name:
                    relaciones.append({
                        "tipo_relacion": rel_label,
                        "cui_relacionado": related_cui,
                        "nombre_relacionado": related_name
                    })
                    
            return relaciones
        except Exception as e:
            print(f"Error obteniendo relaciones de UMLS para {cui}: {e}")
            return []

# Pruebas manuales
if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    try:
        cliente = UMLSClient()
        print("Buscando 'Tenofovir'...")
        resultados = cliente.search_term("Tenofovir")
        for res in resultados[:3]:
            print(f"- {res['cui']}: {res['nombre']}")
    except ValueError as e:
        print(e)
