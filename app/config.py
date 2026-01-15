"""
Configuration module for RAG System.

Centralized configuration for all system components including:
- API endpoints and models
- File paths
- Elasticsearch indices
- Logging settings
"""

import os
from pathlib import Path

# =============================================================================
# API CONFIGURATION
# =============================================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")  # Modelo principal para razonamiento
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")  # Modelo para embeddings
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

# =============================================================================
# FILE PATHS CONFIGURATION
# =============================================================================

# Directorios base
APP_DIR = Path(__file__).parent
PROJECT_ROOT = APP_DIR.parent

# Documentos y datos
DOCUMENT_PATH_PDF_1 = APP_DIR / "pdf" / "PE-3UTE-03318.pdf"  # Documentación técnica 1
DOCUMENT_PATH_PDF_2 = APP_DIR / "pdf" / "PE-3UTE-03319.pdf"  # Documentación técnica 2
DOCUMENT_PATH_TAGS_CSV = APP_DIR / "csv" / "Tags_grupamento.csv"  # Tags para indexar

# =============================================================================
# ELASTICSEARCH INDICES CONFIGURATION
# =============================================================================

INDEX_NAME_TAGS = "tags_index"  # Índice para tags del CSV
INDEX_NAME_GLOBAL_CONTEXT = "global_context_index"  # Índice para documentación PDF

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

# Parámetros del LLM
LLM_TEMPERATURE = 0.1  # Bajo para razonamiento consistente

# Parámetros de chunking
PDF_CHUNK_SIZE = 1500  # Chunks más grandes para contexto técnico
PDF_CHUNK_OVERLAP = 200
PDF_CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# =============================================================================
# API ENDPOINTS CONFIGURATION
# =============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Sistema RAG de Ingeniería de Reservatorios"
API_DESCRIPTION = "Sistema de análisis inteligente para pozos petroleros usando RAG con eventos, tags y documentación técnica"

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# AVAILABLE TAGS FOR MOCK EVENTS
# =============================================================================

AVAILABLE_TAGS = [
    "POCO_MRO_003_SCM_IWIS_MA1_P", "POCO_MRO_003_SCM_IWIS_MA2_P", "POCO_MRO_003_SCM_IWIS_MA3_P",
    "POCO_MRO_003_SCM_IWIS_MA_06_STATUS", "POCO_MRO_003_SCM_IWIS_MA_07_STATUS", "POCO_MRO_003_SCM_IWIS_MA_08_STATUS",
    "POCO_MRO_003_SCM_IWIS_MA1_T", "POCO_MRO_003_SCM_IWIS_MA2_T", "POCO_MRO_003_SCM_IWIS_MA3_T",
    "POCO_MRO_003_SCM_SV_01", "POCO_MRO_003_SCM_SV_02", "POCO_MRO_003_SCM_SV_03", "POCO_MRO_003_SCM_SV_04",
    "POCO_MRO_003_SCM_MA_01", "POCO_MRO_003_SCM_MA_02", "POCO_MRO_003_SCM_MA_03"
]

# =============================================================================
# EVENT TYPES AND SEVERITIES
# =============================================================================

EVENT_TYPES = [
    "Anomalía de Presión", "Pico de Temperatura", "Falla de Sensor", "Cambio de Estado",
    "Alerta de Seguridad", "Variación de Flujo", "Mantenimiento Requerido", "Recuperación de Falla"
]

EVENT_SEVERITIES = ["Low", "Medium", "High", "Critical"]
EVENT_SEVERITY_WEIGHTS = [0.4, 0.4, 0.15, 0.05]  # Más eventos de baja severidad

# =============================================================================
# QUERY PROCESSING CONFIGURATION
# =============================================================================

# Indicadores de preguntas temporales
TIME_INDICATORS = [
    "ayer", "hoy", "último", "última", "últimos", "últimas",
    "pasado", "reciente", "evento", "eventos", "sucede", "sucedió",
    "ocurrió", "ocurrir", "detectó", "detectado", "anomalía", "fallo"
]

# Límites de recuperación
MAX_EVENTS_FOR_QUERY = 5
MAX_TAGS_RESULTS = 3
MAX_RELATED_TAGS = 2
MAX_GLOBAL_DOCS = 4

# =============================================================================
# VECTOR STORE SEARCH CONFIGURATION
# =============================================================================

VECTOR_SEARCH_K = 3
VECTOR_SEARCH_TYPE = "hybrid"