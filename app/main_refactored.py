"""
RAG Application using FastAPI, LangChain, Ollama, and Elasticsearch
Implementa um RAG Híbrido (Semântico + Literal) para Engenharia de Reservatorios.
Arquitectura: Tags indexados localmente + API de Eventos + Contexto Global
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from langchain.prompts import PromptTemplate

from app.config import (
    API_HOST, API_PORT, API_TITLE, API_DESCRIPTION,
    INDEX_NAME_TAGS, INDEX_NAME_GLOBAL_CONTEXT,
    TIME_INDICATORS, MAX_EVENTS_FOR_QUERY, MAX_TAGS_RESULTS, MAX_RELATED_TAGS, MAX_GLOBAL_DOCS
)
from app.models import QueryRequest, QueryResponse, EventsResponse
from app.services.embeddings_service import embeddings_service
from app.services.vector_store_service import vector_store_service
from app.services.events_service import events_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar aplicação FastAPI
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION
)

# Variables globales para servicios
llm_model = None


@app.on_event("startup")
async def startup_event():
    """Inicializar el sistema RAG con arquitectura de dos índices"""
    global llm_model

    try:
        logger.info("Iniciando sistema RAG de Ingeniería de Reservatorios...")

        # Inicializar modelo LLM
        llm_model = embeddings_service.get_llm_model()
        logger.info("Modelo LLM (DeepSeek-R1) inicializado")

        # Inicializar índices (esto puede tomar tiempo si no existen)
        vector_store_service.get_tags_vector_store()
        logger.info("Vector store de tags inicializado")

        vector_store_service.get_global_context_vector_store()
        logger.info("Vector store de contexto global inicializado")

        logger.info("Sistema RAG inicializado exitosamente con arquitectura de dos índices")

    except Exception as e:
        logger.error(f"Falla al inicializar sistema RAG: {str(e)}")
        raise


@app.get("/")
async def root():
    """Endpoint de verificación de salud del sistema"""
    return {
        "message": "Sistema RAG de Ingeniería de Reservatorios",
        "status": "ejecutándose",
        "indices": {
            "tags_index": INDEX_NAME_TAGS,
            "global_context_index": INDEX_NAME_GLOBAL_CONTEXT
        },
        "modelo_llm": "deepseek-r1:14b"
    }


@app.get("/events", response_model=EventsResponse)
async def get_events(
    start_date: str = Query(..., description="Fecha inicio en formato YYYY-MM-DD"),
    end_date: str = Query(..., description="Fecha fin en formato YYYY-MM-DD")
):
    """
    Mock API de Eventos - Simula recuperación de eventos de series de tiempo.

    Esta API reemplaza la funcionalidad de búsqueda de tags por una fuente
    de eventos operativos que impulsan el análisis del sistema RAG.
    """
    try:
        # Validar formato de fechas
        datetime.fromisoformat(start_date + "T00:00:00")
        datetime.fromisoformat(end_date + "T23:59:59")

        # Generar eventos simulados
        events = events_service.generate_mock_events(start_date, end_date)

        logger.info(f"Generados {len(events)} eventos simulados para el período {start_date} - {end_date}")

        return EventsResponse(
            events=events,
            total_events=len(events),
            date_range={"start_date": start_date, "end_date": end_date}
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Formato de fecha inválido: {e}")
    except Exception as e:
        logger.error(f"Error generando eventos: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno: {e}")


@app.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    """
    Endpoint principal de consulta con arquitectura de 4 pasos:
    1. Recuperar Eventos (si pregunta implica tiempo)
    2. Recuperar Tags relacionados a los eventos
    3. Recuperar Contexto Global técnico
    4. Generar respuesta con DeepSeek-R1
    """
    try:
        if llm_model is None:
            raise HTTPException(status_code=500, detail="Sistema RAG no inicializado completamente")

        logger.info(f"Procesando consulta inteligente: {request.question}")

        # =====================================================================
        # PASO 1: RECUPERAR EVENTOS (si la pregunta implica tiempo)
        # =====================================================================
        events_found = []
        question_lower = request.question.lower()

        # Detectar si la pregunta implica análisis temporal
        involves_time = any(indicator in question_lower for indicator in TIME_INDICATORS)

        if involves_time:
            logger.info("Pregunta implica análisis temporal. Consultando eventos...")
            # Usar período de últimos 7 días por defecto
            end_date = datetime.now().date().isoformat()
            start_date = (datetime.now() - timedelta(days=7)).date().isoformat()

            events_response = await get_events(start_date, end_date)
            events_found = events_response.events[:MAX_EVENTS_FOR_QUERY]  # Limitar eventos
            logger.info(f"Recuperados {len(events_found)} eventos relevantes")
        else:
            logger.info("Pregunta no implica tiempo. Saltando recuperación de eventos.")

        # =====================================================================
        # PASO 2: RECUPERAR CONTEXTO DE TAGS
        # =====================================================================
        tags_retrieved = []

        if events_found:
            # Buscar tags exactos de los eventos
            tag_ids_from_events = list(set(event.tag_id for event in events_found))

            logger.info(f"Buscando información técnica de {len(tag_ids_from_events)} tags...")

            # Buscar información de cada tag
            for tag_id in tag_ids_from_events:
                try:
                    # Búsqueda exacta primero
                    exact_results = vector_store_service.similarity_search_with_score(
                        index_name=INDEX_NAME_TAGS,
                        query=f"tag_id:{tag_id}",
                        k=1
                    )

                    if exact_results:
                        doc, score = exact_results[0]
                        tags_retrieved.append({
                            "tag_id": doc.metadata.get("tag_id"),
                            "nomeVariavel": doc.metadata.get("nomeVariavel"),
                            "descricao": doc.metadata.get("descricao"),
                            "grupamento": doc.metadata.get("grupamento"),
                            "sensor_type": doc.metadata.get("sensor_type"),
                            "location": doc.metadata.get("location"),
                            "content": doc.page_content,
                            "search_score": score
                        })

                    # Búsqueda de similitud para tags relacionados
                    related_results = vector_store_service.search_tags(
                        query=f"sensores relacionados a {tag_id}",
                        k=MAX_RELATED_TAGS
                    )

                    for doc in related_results:
                        if doc.metadata.get("tag_id") != tag_id:  # Evitar duplicados
                            tags_retrieved.append({
                                "tag_id": doc.metadata.get("tag_id"),
                                "nomeVariavel": doc.metadata.get("nomeVariavel"),
                                "descricao": doc.metadata.get("descricao"),
                                "grupamento": doc.metadata.get("grupamento"),
                                "sensor_type": doc.metadata.get("sensor_type"),
                                "location": doc.metadata.get("location"),
                                "content": doc.page_content,
                                "relation_type": "related"
                            })

                except Exception as e:
                    logger.warning(f"Error buscando tag {tag_id}: {e}")
                    continue

            logger.info(f"Recuperada información técnica de {len(tags_retrieved)} tags")

        # =====================================================================
        # PASO 3: RECUPERAR CONTEXTO GLOBAL
        # =====================================================================
        logger.info("Recuperando contexto global de documentación técnica...")

        # Crear query para contexto global basada en eventos y tags
        global_query_parts = [request.question]

        if events_found:
            event_types = list(set(event.type for event in events_found))
            global_query_parts.append(f"explicación técnica de: {' '.join(event_types)}")

        if tags_retrieved:
            sensor_types = list(set(tag.get("sensor_type", "") for tag in tags_retrieved if tag.get("sensor_type")))
            if sensor_types:
                global_query_parts.append(f"funcionamiento de: {' '.join(sensor_types)}")

        global_query = " ".join(global_query_parts)

        logger.info(f"Query para contexto global: {global_query}")

        global_docs = vector_store_service.search_global_context(
            query=global_query,
            k=MAX_GLOBAL_DOCS
        )
        logger.info(f"Recuperados {len(global_docs)} documentos de contexto global")

        # =====================================================================
        # PASO 4: GENERACIÓN CON DEEPSEEK-R1
        # =====================================================================

        # Preparar contexto completo para el LLM
        context_parts = []

        # Eventos
        if events_found:
            context_parts.append("EVENTOS DETECTADOS:")
            for event in events_found:
                context_parts.append(f"- Evento {event.event_id}: {event.description} (Tag: {event.tag_id}, Severidad: {event.severity})")

        # Tags técnicos
        if tags_retrieved:
            context_parts.append("\nSENSORES TÉCNICOS INVOLUCRADOS:")
            for tag in tags_retrieved:
                context_parts.append(f"- {tag['nomeVariavel']}: {tag['descricao']} (Tipo: {tag.get('sensor_type', 'N/A')}, Ubicación: {tag.get('location', 'N/A')})")

        # Documentación técnica
        if global_docs:
            context_parts.append("\nDOCUMENTACIÓN TÉCNICA:")
            for i, doc in enumerate(global_docs):
                context_parts.append(f"Documento {i+1}: {doc.page_content[:500]}...")

        full_context = "\n".join(context_parts)

        # Prompt para DeepSeek-R1 con Chain of Thought
        cot_prompt_template = """
Eres un Analista Senior de Reservorios y Pozos Inteligentes.
Tu objetivo es analizar eventos operativos basándote en evidencia de datos y documentación técnica.

PASO 1: Analiza los EVENTOS DETECTADOS (Series temporales).
PASO 2: Analiza los SENSORES (TAGS) involucrados y sus relaciones físicas.
PASO 3: Consulta la DOCUMENTACIÓN (Contexto Global) para explicar el fenómeno.

Contexto Recuperado:
{context}

Pregunta del Usuario: {question}

Proceso de Pensamiento (Chain of Thought):
1. Identificar qué anomalía ocurrió y cuándo.
2. Relacionar el tag afectado con su función en el pozo.
3. Buscar posibles causas en la documentación.
4. Concluir con una recomendación técnica.

Respuesta Final:
"""

        cot_prompt = PromptTemplate(
            template=cot_prompt_template,
            input_variables=["context", "question"]
        )

        # Generar respuesta final
        chain = cot_prompt | llm_model
        final_answer = chain.invoke({
            "context": full_context,
            "question": request.question
        })

        # Preparar documentos fuente para respuesta
        source_documents = []

        # Agregar eventos como fuentes
        for event in events_found:
            source_documents.append({
                "content": f"Evento {event.event_id}: {event.description}",
                "metadata": {
                    "source": "events_api",
                    "type": "time_series_event",
                    "event_id": event.event_id,
                    "tag_id": event.tag_id,
                    "timestamp": event.timestamp,
                    "severity": event.severity
                }
            })

        # Agregar tags como fuentes
        for tag in tags_retrieved:
            source_documents.append({
                "content": tag.get("content", ""),
                "metadata": {
                    "source": "tags_index",
                    "type": "sensor_technical_info",
                    "tag_id": tag.get("tag_id"),
                    "sensor_type": tag.get("sensor_type"),
                    "location": tag.get("location")
                }
            })

        # Agregar documentación global como fuentes
        for doc in global_docs:
            source_documents.append({
                "content": doc.page_content,
                "metadata": {
                    "source": doc.metadata.get("source", "global_context"),
                    "type": "technical_documentation",
                    "chunk_id": doc.metadata.get("chunk_id")
                }
            })

        logger.info(f"Consulta procesada exitosamente. Respuesta generada con {len(source_documents)} fuentes")

        return QueryResponse(
            answer=str(final_answer),
            source_documents=source_documents,
            events_found=[event.dict() for event in events_found],
            tags_retrieved=tags_retrieved
        )

    except Exception as e:
        logger.error(f"Error procesando consulta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Nota: O uvicorn.run() aquí es apenas para desenvolvimento.
    # Em produção, use o entrypoint do Docker Compose.
    uvicorn.run(app, host=API_HOST, port=API_PORT)