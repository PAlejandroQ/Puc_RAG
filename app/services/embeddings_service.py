"""
Embeddings Service for RAG System.

Handles creation and management of embedding models and LLMs using Ollama.
"""

import logging
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_EMBEDDING_MODEL,
    LLM_TEMPERATURE
)

logger = logging.getLogger(__name__)


class EmbeddingsService:
    """Servicio para gestión de modelos de embeddings y LLM"""

    def __init__(self):
        self._embeddings_model = None
        self._llm_model = None

    def get_embeddings_model(self) -> OllamaEmbeddings:
        """Crear o retornar modelo de embeddings con configuración específica"""
        if self._embeddings_model is None:
            self._embeddings_model = OllamaEmbeddings(
                model=OLLAMA_EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )
            logger.info(f"Modelo de embeddings inicializado: {OLLAMA_EMBEDDING_MODEL}")
        return self._embeddings_model

    def get_llm_model(self) -> Ollama:
        """Crear o retornar modelo LLM principal (DeepSeek-R1)"""
        if self._llm_model is None:
            self._llm_model = Ollama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_BASE_URL,
                temperature=LLM_TEMPERATURE  # Bajo para razonamiento consistente
            )
            logger.info(f"Modelo LLM inicializado: {OLLAMA_MODEL}")
        return self._llm_model

    def reset_models(self):
        """Reiniciar modelos (útil para testing)"""
        self._embeddings_model = None
        self._llm_model = None
        logger.info("Modelos reiniciados")


# Instancia global del servicio
embeddings_service = EmbeddingsService()