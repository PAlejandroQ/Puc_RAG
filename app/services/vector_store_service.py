"""
Vector Store Service for RAG System.

Handles Elasticsearch vector stores for tags and global context documents.
"""

import logging
from pathlib import Path
from typing import List, Optional

from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from elasticsearch import Elasticsearch

from app.config import (
    ELASTICSEARCH_URL,
    INDEX_NAME_TAGS,
    INDEX_NAME_GLOBAL_CONTEXT,
    DOCUMENT_PATH_TAGS_CSV,
    DOCUMENT_PATH_PDF_1,
    DOCUMENT_PATH_PDF_2,
    VECTOR_SEARCH_K,
    VECTOR_SEARCH_TYPE
)
from app.services.embeddings_service import embeddings_service
from app.utils.tag_processing import ingest_tags_to_elasticsearch
from app.utils.document_processing import load_global_context_documents

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Servicio para gestión de vector stores en Elasticsearch"""

    def __init__(self):
        self._tags_vector_store: Optional[ElasticsearchStore] = None
        self._global_context_vector_store: Optional[ElasticsearchStore] = None
        self._es_client = Elasticsearch(ELASTICSEARCH_URL)

    def get_tags_vector_store(self) -> ElasticsearchStore:
        """Obtener o inicializar vector store de tags"""
        if self._tags_vector_store is None:
            self._tags_vector_store = self._initialize_tags_vector_store()
        return self._tags_vector_store

    def get_global_context_vector_store(self) -> ElasticsearchStore:
        """Obtener o inicializar vector store de contexto global"""
        if self._global_context_vector_store is None:
            self._global_context_vector_store = self._initialize_global_context_vector_store()
        return self._global_context_vector_store

    def _initialize_tags_vector_store(self) -> ElasticsearchStore:
        """
        Inicializa el vector store para tags indexados localmente.
        """
        try:
            embeddings = embeddings_service.get_embeddings_model()

            if self._es_client.indices.exists(index=INDEX_NAME_TAGS):
                logger.info(f"Índice de tags '{INDEX_NAME_TAGS}' encontrado. Cargando...")
                return ElasticsearchStore(
                    es_url=ELASTICSEARCH_URL,
                    index_name=INDEX_NAME_TAGS,
                    embedding=embeddings
                )
            else:
                logger.info(f"Índice de tags '{INDEX_NAME_TAGS}' no encontrado. Creando nuevo...")

                if not Path(DOCUMENT_PATH_TAGS_CSV).exists():
                    raise FileNotFoundError(f"CSV de tags no encontrado: {DOCUMENT_PATH_TAGS_CSV}")

                # Procesar e indexar tags
                tag_documents = ingest_tags_to_elasticsearch(DOCUMENT_PATH_TAGS_CSV)

                if tag_documents:
                    logger.info(f"Indexando {len(tag_documents)} tags en Elasticsearch...")
                    vector_store = ElasticsearchStore.from_documents(
                        documents=tag_documents,
                        embedding=embeddings,
                        es_url=ELASTICSEARCH_URL,
                        index_name=INDEX_NAME_TAGS
                    )
                    logger.info("Vector store de tags creado exitosamente")
                    return vector_store
                else:
                    raise ValueError("No se pudieron procesar tags para indexar")

        except Exception as e:
            logger.error(f"Error inicializando vector store de tags: {e}")
            raise

    def _initialize_global_context_vector_store(self) -> ElasticsearchStore:
        """
        Inicializa el vector store para contexto global (documentación técnica).
        """
        try:
            embeddings = embeddings_service.get_embeddings_model()

            if self._es_client.indices.exists(index=INDEX_NAME_GLOBAL_CONTEXT):
                logger.info(f"Índice de contexto global '{INDEX_NAME_GLOBAL_CONTEXT}' encontrado. Cargando...")
                return ElasticsearchStore(
                    es_url=ELASTICSEARCH_URL,
                    index_name=INDEX_NAME_GLOBAL_CONTEXT,
                    embedding=embeddings
                )
            else:
                logger.info(f"Índice de contexto global '{INDEX_NAME_GLOBAL_CONTEXT}' no encontrado. Creando nuevo...")

                # Procesar documentación PDF
                context_documents = load_global_context_documents()

                if context_documents:
                    logger.info(f"Indexando {len(context_documents)} chunks de documentación...")
                    vector_store = ElasticsearchStore.from_documents(
                        documents=context_documents,
                        embedding=embeddings,
                        es_url=ELASTICSEARCH_URL,
                        index_name=INDEX_NAME_GLOBAL_CONTEXT
                    )
                    logger.info("Vector store de contexto global creado exitosamente")
                    return vector_store
                else:
                    # Crear documentación por defecto si no hay PDFs
                    logger.warning("No se encontraron documentos PDF. Creando contexto por defecto...")
                    default_docs = [Document(
                        page_content="""
                        Este es un sistema de análisis inteligente para pozos petroleros.
                        Proporciona información técnica sobre sensores, válvulas, presiones,
                        temperaturas y flujos en operaciones de reservorios petroleros.
                        Incluye monitoreo de PDGs (Permanent Downhole Gauges), ICVs (Interval Control Valves),
                        y sistemas SCM (Smart Completion Systems).
                        """,
                        metadata={"source": "default", "data_type": "technical_documentation"}
                    )]
                    vector_store = ElasticsearchStore.from_documents(
                        documents=default_docs,
                        embedding=embeddings,
                        es_url=ELASTICSEARCH_URL,
                        index_name=INDEX_NAME_GLOBAL_CONTEXT
                    )
                    return vector_store

        except Exception as e:
            logger.error(f"Error inicializando vector store de contexto global: {e}")
            raise

    def search_tags(self, query: str, k: int = VECTOR_SEARCH_K, search_type: str = VECTOR_SEARCH_TYPE):
        """Buscar en el índice de tags"""
        retriever = self.get_tags_vector_store().as_retriever(
            search_kwargs={"k": k, "search_type": search_type}
        )
        return retriever.get_relevant_documents(query)

    def search_global_context(self, query: str, k: int = VECTOR_SEARCH_K, search_type: str = VECTOR_SEARCH_TYPE):
        """Buscar en el índice de contexto global"""
        retriever = self.get_global_context_vector_store().as_retriever(
            search_kwargs={"k": k, "search_type": search_type}
        )
        return retriever.get_relevant_documents(query)

    def similarity_search_with_score(self, index_name: str, query: str, k: int = 1):
        """Búsqueda de similitud con score"""
        if index_name == INDEX_NAME_TAGS:
            vector_store = self.get_tags_vector_store()
        elif index_name == INDEX_NAME_GLOBAL_CONTEXT:
            vector_store = self.get_global_context_vector_store()
        else:
            raise ValueError(f"Índice desconocido: {index_name}")

        return vector_store.similarity_search_with_score(query, k=k)

    def reset_stores(self):
        """Reiniciar vector stores (útil para testing)"""
        self._tags_vector_store = None
        self._global_context_vector_store = None
        logger.info("Vector stores reiniciados")


# Instancia global del servicio
vector_store_service = VectorStoreService()