"""
API Mock para consulta de descripciones de TAGs de ingeniería.
Lee datos desde Tags_grupamento.csv y soporta búsqueda por regex.
"""

import re
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar aplicación FastAPI
app = FastAPI(title="API de TAGs", description="API mock para consulta de descripciones de TAGs usando regex")

# Modelo de respuesta
class TagInfo(BaseModel):
    nomeVariavel: str
    descricao: str
    grupamento: Optional[str] = None

class TagResponse(BaseModel):
    tags: List[TagInfo]
    total_matches: int

# Cargar datos del CSV al iniciar
TAGS_DATA = None

def load_tags_data():
    """Carga los datos de tags desde el CSV"""
    global TAGS_DATA
    try:
        csv_path = "app/csv/Tags_grupamento.csv"
        df = pd.read_csv(csv_path)
        TAGS_DATA = df.to_dict('records')
        logger.info(f"Datos de tags cargados: {len(TAGS_DATA)} registros desde {csv_path}")
        return TAGS_DATA
    except Exception as e:
        logger.error(f"Error al cargar datos del CSV: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Cargar datos al iniciar la aplicación"""
    load_tags_data()

@app.get("/")
async def root():
    """Endpoint de verificación de salud"""
    return {"message": "API de TAGs ejecutándose", "status": "OK", "total_tags": len(TAGS_DATA) if TAGS_DATA else 0}

@app.get("/tags/search", response_model=TagResponse)
async def search_tags(
    tag_pattern: str = Query(..., description="Patrón de búsqueda para TAGs (soporta regex). Ej: 'POCO_MRO_003.*STATUS' o 'ICV'"),
    limit: Optional[int] = Query(50, description="Número máximo de resultados a retornar")
):
    """
    Busca tags usando regex pattern matching.

    El parámetro tag_pattern se trata como una expresión regular de Python.
    Puede ser un match exacto o un patrón más amplio.

    Ejemplos:
    - Match exacto: "POCO_MRO_003_SCM_IWIS_MA_06_STATUS"
    - Patrón con wildcards: "POCO_MRO_003.*STATUS"
    - Búsqueda por tipo: ".*ICV.*"
    - Búsqueda por zona: ".*INFERIOR.*"
    """
    if not TAGS_DATA:
        raise HTTPException(status_code=500, detail="Datos de tags no disponibles")

    try:
        # Compilar el patrón regex
        pattern = re.compile(tag_pattern, re.IGNORECASE)
        matches = []

        # Buscar matches en los datos
        for tag_data in TAGS_DATA:
            nome_variavel = tag_data.get('nomeVariavel', '')
            if pattern.search(nome_variavel):
                matches.append(TagInfo(
                    nomeVariavel=nome_variavel,
                    descricao=tag_data.get('descricao', ''),
                    grupamento=tag_data.get('grupamento', None)
                ))

        # Limitar resultados si se especifica
        if limit and len(matches) > limit:
            matches = matches[:limit]

        logger.info(f"Búsqueda regex '{tag_pattern}' encontró {len(matches)} matches")

        return TagResponse(
            tags=matches,
            total_matches=len(matches)
        )

    except re.error as e:
        logger.error(f"Error en patrón regex '{tag_pattern}': {e}")
        raise HTTPException(status_code=400, detail=f"Patrón regex inválido: {e}")
    except Exception as e:
        logger.error(f"Error al procesar búsqueda: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")

@app.get("/tags/{tag_name}", response_model=TagInfo)
async def get_tag_by_name(tag_name: str):
    """
    Obtiene información de un tag específico por nombre exacto.
    """
    if not TAGS_DATA:
        raise HTTPException(status_code=500, detail="Datos de tags no disponibles")

    for tag_data in TAGS_DATA:
        if tag_data.get('nomeVariavel') == tag_name:
            return TagInfo(
                nomeVariavel=tag_name,
                descricao=tag_data.get('descricao', ''),
                grupamento=tag_data.get('grupamento', None)
            )

    raise HTTPException(status_code=404, detail=f"Tag '{tag_name}' no encontrado")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)