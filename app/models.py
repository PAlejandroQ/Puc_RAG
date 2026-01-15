"""
Data models for RAG System.

Pydantic models for API requests, responses, and internal data structures.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


# =============================================================================
# API REQUEST/RESPONSE MODELS
# =============================================================================

class QueryRequest(BaseModel):
    """Modelo de requisição para o endpoint de consulta"""
    question: str


class QueryResponse(BaseModel):
    """Modelo de resposta para o endpoint de consulta"""
    answer: str
    source_documents: List[Dict[str, Any]]
    events_found: Optional[List[Dict[str, Any]]] = None
    tags_retrieved: Optional[List[Dict[str, Any]]] = None


class TagInfo(BaseModel):
    """Modelo para información de tags indexados"""
    tag_id: str
    nomeVariavel: str
    descricao: str
    grupamento: Optional[str] = None
    unidade: Optional[str] = None


class EventInfo(BaseModel):
    """Modelo para eventos de series de tiempo"""
    event_id: int
    tag_id: str
    timestamp: str
    type: str
    severity: str
    description: str


class EventsResponse(BaseModel):
    """Modelo de respuesta de la API de eventos"""
    events: List[EventInfo]
    total_events: int
    date_range: Dict[str, str]


# =============================================================================
# INTERNAL DATA MODELS
# =============================================================================

class TagComponents(BaseModel):
    """Modelo para componentes parseados de un tag"""
    componente_1: Optional[str] = None
    componente_2: Optional[str] = None
    componente_3: Optional[str] = None


class ProcessedTag(BaseModel):
    """Modelo para tag procesado con metadatos completos"""
    tag_id: str
    nomeVariavel: str
    descricao: str
    grupamento: Optional[str] = None
    unidade: Optional[str] = None
    sensor_type: str
    location: str
    function: str
    descriptive_content: str


class DocumentMetadata(BaseModel):
    """Modelo para metadatos de documentos indexados"""
    source: str
    data_type: str
    tag_id: Optional[str] = None
    sensor_type: Optional[str] = None
    location: Optional[str] = None
    chunk_id: Optional[int] = None
    document_type: Optional[str] = None
    section_id: Optional[int] = None
    topic: Optional[str] = None


class VectorSearchResult(BaseModel):
    """Modelo para resultados de búsqueda vectorial"""
    content: str
    metadata: DocumentMetadata
    score: Optional[float] = None
    relation_type: Optional[str] = None


class QueryContext(BaseModel):
    """Modelo para contexto completo de una consulta"""
    question: str
    events: List[EventInfo] = []
    tags: List[VectorSearchResult] = []
    global_docs: List[VectorSearchResult] = []
    full_context: str = ""


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class ElasticsearchConfig(BaseModel):
    """Configuración para Elasticsearch"""
    url: str
    tags_index: str
    global_context_index: str


class OllamaConfig(BaseModel):
    """Configuración para Ollama"""
    base_url: str
    model: str
    embedding_model: str
    temperature: float


class SystemConfig(BaseModel):
    """Configuración completa del sistema"""
    elasticsearch: ElasticsearchConfig
    ollama: OllamaConfig
    api_host: str
    api_port: int