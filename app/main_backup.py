"""
RAG Application using FastAPI, LangChain, Ollama, and Elasticsearch
Implementa um RAG Híbrido (Semântico + Literal) para Engenharia de Reservatorios.
Arquitectura: Tags indexados localmente + API de Eventos + Contexto Global
"""

import os
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import random

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from elasticsearch import Elasticsearch
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar aplicação FastAPI
app = FastAPI(
    title="Sistema RAG de Engenharia de Reservatorios",
    description="Sistema de análisis inteligente para pozos petroleros usando RAG con eventos, tags y documentación técnica"
)

# Configuração - Arquitectura de dos índices separados
DOCUMENT_PATH_PDF_1 = "app/pdf/PE-3UTE-03318.pdf"  # Documentación técnica 1
DOCUMENT_PATH_PDF_2 = "app/pdf/PE-3UTE-03319.pdf"  # Documentación técnica 2
DOCUMENT_PATH_TAGS_CSV = "app/csv/Tags_grupamento.csv"  # Tags para indexar

INDEX_NAME_TAGS = "tags_index"  # Índice para tags del CSV
INDEX_NAME_GLOBAL_CONTEXT = "global_context_index"  # Índice para documentación PDF

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")  # Modelo principal para razonamiento
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")  # Modelo para embeddings
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")

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
# FUNCIONES PARA GESTIÓN DE ÍNDICES Y DATOS
# =============================================================================

def create_embeddings_model() -> OllamaEmbeddings:
    """Crear modelo de embeddings con configuración específica"""
    return OllamaEmbeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

def create_llm_model() -> Ollama:
    """Crear modelo LLM principal (DeepSeek-R1)"""
    return Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1  # Bajo para razonamiento consistente
    )

def ingest_tags_to_elasticsearch(csv_path: str) -> List[Document]:
    """
    Lee el CSV de tags y crea documentos para indexar en Elasticsearch.

    Args:
        csv_path: Ruta al archivo CSV de tags

    Returns:
        Lista de documentos LangChain para indexar
    """
    logger.info(f"Procesando tags desde {csv_path}...")

    try:
        df = pd.read_csv(csv_path)
        documents = []

        for _, row in df.iterrows():
            tag_id = row.get('nomeVariavel', '').strip()
            if not tag_id or tag_id == 'ND':
                continue

            # Crear contenido para embedding (descripción técnica)
            description = row.get('descricao', 'Sin descripción')
            grupamento = row.get('grupamento', 'Sin grupo')

            # Contenido rico para embedding semántico
            content = f"""
Tag: {tag_id}
Descripción: {description}
Grupo: {grupamento}
Tipo de Sensor: {infer_sensor_type(tag_id, description)}
Ubicación: {infer_location(tag_id)}
Función: {infer_function(description)}
""".strip()

            metadata = {
                "tag_id": tag_id,
                "nomeVariavel": tag_id,
                "descricao": description,
                "grupamento": grupamento,
                "sensor_type": infer_sensor_type(tag_id, description),
                "location": infer_location(tag_id),
                "data_type": "tag_sensor"
            }

            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)

        logger.info(f"Procesados {len(documents)} tags para indexación")
        return documents

    except Exception as e:
        logger.error(f"Error procesando CSV de tags: {e}")
        raise

def infer_sensor_type(tag_name: str, description: str) -> str:
    """Infere el tipo de sensor basado en el nombre del tag y descripción"""
    tag_lower = tag_name.lower()
    desc_lower = description.lower()

    if 'pressao' in desc_lower or 'pressão' in desc_lower or 'ma' in tag_lower:
        return "Sensor de Presión"
    elif 'temperatura' in desc_lower or 'temperature' in desc_lower:
        return "Sensor de Temperatura"
    elif 'vazao' in desc_lower or 'flow' in desc_lower or 'vz' in tag_lower:
        return "Sensor de Flujo"
    elif 'status' in desc_lower or 'sv' in tag_lower:
        return "Sensor de Estado/Posición"
    elif 'pdg' in tag_lower:
        return "Permanent Downhole Gauge"
    elif 'icv' in tag_lower:
        return "Interval Control Valve"
    else:
        return "Sensor Genérico"

def infer_location(tag_name: str) -> str:
    """Infere la ubicación del sensor basado en el nombre del tag"""
    tag_lower = tag_name.lower()

    if 'anular' in tag_lower or 'annulus' in tag_lower:
        return "Anular del Pozo"
    elif 'coluna' in tag_lower or 'column' in tag_lower:
        return "Columna de Producción"
    elif 'cabeça' in tag_lower or 'head' in tag_lower:
        return "Cabeza del Pozo"
    elif 'scm' in tag_lower:
        return "Sistema de Control de Pozo"
    else:
        return "Ubicación Genérica"

def infer_function(description: str) -> str:
    """Infere la función del sensor basada en la descripción"""
    desc_lower = description.lower()

    if 'controle' in desc_lower or 'control' in desc_lower:
        return "Control de Producción"
    elif 'monitor' in desc_lower or 'monitoring' in desc_lower:
        return "Monitoreo de Condiciones"
    elif 'status' in desc_lower or 'estado' in desc_lower:
        return "Estado del Equipo"
    elif 'segurança' in desc_lower or 'safety' in desc_lower:
        return "Seguridad del Pozo"
    else:
        return "Medición Operacional"

def load_global_context_documents() -> List[Document]:
    """
    Carga y procesa la documentación PDF para el contexto global.

    Returns:
        Lista de documentos chunked para indexar
    """
    documents = []

    pdf_paths = [DOCUMENT_PATH_PDF_1, DOCUMENT_PATH_PDF_2]

    for pdf_path in pdf_paths:
        if Path(pdf_path).exists():
            logger.info(f"Cargando documentación desde {pdf_path}...")
            try:
                loader = PyPDFLoader(pdf_path)
                pdf_docs = loader.load()

                # Chunking apropiado para documentos técnicos largos
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,  # Chunks más grandes para contexto técnico
                    chunk_overlap=200,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )

                chunks = text_splitter.split_documents(pdf_docs)

                # Agregar metadata específica
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "source": pdf_path,
                        "chunk_id": i,
                        "data_type": "technical_documentation",
                        "document_type": "reservoir_engineering_manual"
                    })

                documents.extend(chunks)
                logger.info(f"Procesados {len(chunks)} chunks desde {pdf_path}")

            except Exception as e:
                logger.error(f"Error cargando {pdf_path}: {e}")
        else:
            logger.warning(f"Documento no encontrado: {pdf_path}")

    return documents

# =============================================================================
# MOCK API DE EVENTOS (Reemplaza la API de tags)
# =============================================================================

def generate_mock_events(start_date: str, end_date: str) -> List[EventInfo]:
    """
    Genera eventos simulados de series de tiempo para el período solicitado.

    Args:
        start_date: Fecha inicio en formato YYYY-MM-DD
        end_date: Fecha fin en formato YYYY-MM-DD

    Returns:
        Lista de eventos simulados
    """
    # Tags disponibles para generar eventos
    available_tags = [
        "POCO_MRO_003_SCM_IWIS_MA1_P", "POCO_MRO_003_SCM_IWIS_MA2_P", "POCO_MRO_003_SCM_IWIS_MA3_P",
        "POCO_MRO_003_SCM_IWIS_MA_06_STATUS", "POCO_MRO_003_SCM_IWIS_MA_07_STATUS", "POCO_MRO_003_SCM_IWIS_MA_08_STATUS",
        "POCO_MRO_003_SCM_IWIS_MA1_T", "POCO_MRO_003_SCM_IWIS_MA2_T", "POCO_MRO_003_SCM_IWIS_MA3_T",
        "POCO_MRO_003_SCM_SV_01", "POCO_MRO_003_SCM_SV_02", "POCO_MRO_003_SCM_SV_03", "POCO_MRO_003_SCM_SV_04",
        "POCO_MRO_003_SCM_MA_01", "POCO_MRO_003_SCM_MA_02", "POCO_MRO_003_SCM_MA_03"
    ]

    # Tipos de eventos posibles
    event_types = [
        "Anomalía de Presión", "Pico de Temperatura", "Falla de Sensor", "Cambio de Estado",
        "Alerta de Seguridad", "Variación de Flujo", "Mantenimiento Requerido", "Recuperación de Falla"
    ]

    severities = ["Low", "Medium", "High", "Critical"]

    events = []
    event_id = 1000

    # Convertir fechas
    start = datetime.fromisoformat(start_date + "T00:00:00")
    end = datetime.fromisoformat(end_date + "T23:59:59")

    # Generar eventos aleatorios (entre 3-8 eventos por consulta)
    num_events = random.randint(3, 8)

    for i in range(num_events):
        # Timestamp aleatorio en el rango
        time_diff = (end - start).total_seconds()
        random_seconds = random.randint(0, int(time_diff))
        event_time = start + timedelta(seconds=random_seconds)

        # Seleccionar tag aleatorio
        tag_id = random.choice(available_tags)

        # Seleccionar tipo y severidad
        event_type = random.choice(event_types)
        severity = random.choice(severities)

        # Generar descripción basada en el tipo de evento
        descriptions = {
            "Anomalía de Presión": [
                f"Detectada anomalía de presión en {tag_id}. Valor fuera de rango operativo.",
                f"Pico de presión registrado en sensor {tag_id}. Posible obstrucción.",
                f"Caída brusca de presión en {tag_id}. Revisar integridad del sistema."
            ],
            "Pico de Temperatura": [
                f"Temperatura elevada detectada en {tag_id}. Verificar refrigeración.",
                f"Alerta térmica en sensor {tag_id}. Temperatura por encima del límite.",
                f"Sobrecarga térmica registrada en {tag_id}."
            ],
            "Falla de Sensor": [
                f"Falla de comunicación detectada en sensor {tag_id}.",
                f"Pérdida de señal en {tag_id}. Requiere mantenimiento.",
                f"Error de calibración en sensor {tag_id}."
            ],
            "Cambio de Estado": [
                f"Cambio de estado registrado en válvula {tag_id}.",
                f"Activación automática de sistema de seguridad en {tag_id}.",
                f"Transición de estado detectada en {tag_id}."
            ],
            "Alerta de Seguridad": [
                f"Condición de seguridad activada en {tag_id}.",
                f"Alerta crítica de sistema en {tag_id}. Acción requerida.",
                f"Protocolo de seguridad iniciado por sensor {tag_id}."
            ],
            "Variación de Flujo": [
                f"Variación significativa de flujo detectada en {tag_id}.",
                f"Cambio en patrón de flujo registrado por {tag_id}.",
                f"Anomalía de caudal en sensor {tag_id}."
            ],
            "Mantenimiento Requerido": [
                f"Mantenimiento preventivo recomendado para {tag_id}.",
                f"Alerta de desgaste en componente de {tag_id}.",
                f"Revisión programada requerida en sensor {tag_id}."
            ],
            "Recuperación de Falla": [
                f"Recuperación automática exitosa en {tag_id}.",
                f"Sistema restablecido en sensor {tag_id}.",
                f"Normalización de condiciones en {tag_id}."
            ]
        }

        description = random.choice(descriptions.get(event_type, ["Evento registrado en sensor."]))

        event = EventInfo(
            event_id=event_id + i,
            tag_id=tag_id,
            timestamp=event_time.isoformat(),
            type=event_type,
            severity=severity,
            description=description
        )

        events.append(event)

    # Ordenar por timestamp
    events.sort(key=lambda x: x.timestamp)

    return events

def parse_tag(tag_string: str) -> Optional[Dict[str, str]]:
    """
    Divide um TAG literal nos componentes identificados usando regex.
    Suporta múltiplos formatos:

    Formato 1 (3 componentes): "PI 7-MRO-3-RJS PDG Coluna acima do Packer"
    - Componente 1: (Tipo) Sem espaço
    - Componente 2: (ID Poço) Sem espaço
    - Componente 3: (Descrição/Local) Restante da string

    Formato 2 (2 componentes POCO): "POCO_MRO_003_SCM_IWIS_MA_06_STATUS"
    - Componente 1: POCO_MRO_XXX (onde XXX são números)
    - Componente 2: Restante do TAG
    - Componente 3: None

    Formato 3 (2 componentes simples): "TIPO Descrição"

    Exemplos:
    - "PI 7-MRO-3-RJS PDG Coluna acima do Packer" → {'componente_1': 'PI', 'componente_2': '7-MRO-3-RJS', 'componente_3': 'PDG Coluna acima do Packer'}
    - "POCO_MRO_003_SCM_IWIS_MA_06_STATUS" → {'componente_1': 'POCO_MRO_003', 'componente_2': 'SCM_IWIS_MA_06_STATUS', 'componente_3': None}
    """
    if not isinstance(tag_string, str):
        return None
        
    # Regex: ^(\S+) \s+ (\S+) \s+ (.+)
    # \S+ = Pelo menos um caractere sem espaço (Componente 1 e 2)
    # \s+ = Pelo menos um espaço (separadores)
    # .+  = Todos os caracteres restantes (Componente 3)
    match = re.match(r"^(\S+)\s+(\S+)\s+(.+)$", tag_string)
    
    if match:
        return {
            "componente_1": match.group(1),
            "componente_2": match.group(2),
            "componente_3": match.group(3)
        }
    else:
        # Tentar um padrão para tags como POCO_MRO_003_SCM_IWIS_MA_06_STATUS
        # Onde componente 1 é POCO_MRO_XXX (XXX sendo números) e componente 2 é o resto
        match_poco = re.match(r"^(POCO_MRO_\d+)_(.+)$", tag_string)
        if match_poco:
            return {
                "componente_1": match_poco.group(1),  # POCO_MRO_003
                "componente_2": match_poco.group(2),  # SCM_IWIS_MA_06_STATUS
                "componente_3": None  # No hay componente 3 en este formato
            }
        else:
            # Tentar um padrão mais simples se o primeiro falhar (ex: TAGs com apenas 2 componentes)
            match_simple = re.match(r"^(\S+)\s+(.+)$", tag_string)
            if match_simple:
                 return {
                    "componente_1": match_simple.group(1),
                    "componente_2": None, # Não corresponde ao padrão de 3
                    "componente_3": match_simple.group(2)
                }
        logger.warning(f"TAG '{tag_string}' não correspondeu a nenhum padrão de parse.")
        return None

def generate_descriptive_context(llm: Ollama, components: Dict[str, str], row: Dict[str, Any], general_context: str = "") -> str:
    """
    Usa o LLM configurado para gerar um "Contexto Descritivo" rico para o embedding.
    Combina os componentes do TAG com os dados originais do CSV e usa o contexto geral para enriquecer.
    """
    
    # Template do prompt para o LLM de geração
    generation_template = """Você é um especialista em engenharia de petróleo e instrumentação (Oil & Gas).
Sua tarefa é gerar uma descrição técnica unificada e rica em contexto para um TAG de medição, com base nas informações fornecidas e no contexto geral dos especialistas.
Esta descrição será usada para busca semântica (embedding), então deve ser completa e explicar siglas quando necessário.

**Contexto Geral dos Especialistas:**
{general_context}

**Informações Fornecidas:**
- **TAG Literal:** {tag_literal}
- **Descrição Original:** {descricao}
- **Grupamento:** {grupamento}
- **Componente 1 (Tipo):** {componente_1}
- **Componente 2 (ID Poço/Sistema):** {componente_2}
- **Componente 3 (Medição/Local):** {componente_3}

**Instrução:**
Gere um parágrafo conciso (em português) que descreva este ponto de medição. Use o contexto geral para explicar siglas e termos técnicos. Combine todas as informações de forma lógica, incluindo o grupamento quando disponível.
Exemplo de Resposta: "Este é um ponto de medição do poço POCO_MRO_003, pertencente ao grupamento 'PDG Status e Pressões no interior e cabeça do poço'. O componente MA_06_STATUS indica o status de um Medidor Permanente de Fundo de Poço (PDG) na zona intermediária da coluna de produção."

**Contexto Descritivo Unificado (para embedding):**
"""
    
    prompt = PromptTemplate(
        template=generation_template,
        input_variables=[
            "general_context", "tag_literal", "descricao", "grupamento",
            "componente_1", "componente_2", "componente_3"
        ]
    )

    # Preencher com os dados da linha, tratando valores ausentes
    input_data = {
        "general_context": general_context,
        "tag_literal": row.get("nomeVariavel", "N/A"),
        "descricao": row.get("descricao", "N/A"),
        "grupamento": row.get("grupamento", "N/A"),
        "componente_1": components.get("componente_1", "N/A"),
        "componente_2": components.get("componente_2", "N/A"),
        "componente_3": components.get("componente_3", "N/A")
    }

    try:
        # Criar a cadeia e invocar o LLM
        chain = prompt | llm
        descriptive_context = chain.invoke(input_data)
        return descriptive_context.strip()
    except Exception as e:
        logger.error(f"Erro ao gerar contexto para o TAG '{row.get('nomeVariavel')}': {e}")
        # Retornar uma descrição de fallback caso o LLM falhe
        return f"TAG: {row.get('nomeVariavel', 'N/A')}. Descrição: {row.get('descricao', 'N/A')}. Grupamento: {row.get('grupamento', 'N/A')}."


# =============================================================================
# FUNÇÕES PRINCIPAIS (Modificadas)
# =============================================================================

def create_embeddings_model() -> OllamaEmbeddings:
    """Criar instância do modelo de embeddings."""
    return OllamaEmbeddings(
        model=OLLAMA_MODEL, # Usar el modelo configurado
        base_url=OLLAMA_BASE_URL
    )

def create_llm_model() -> Ollama:
    """Criar instância do LLM."""
    return Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )

def load_general_context(file_path: str) -> List[Document]:
    """
    Carrega o contexto geral do arquivo de texto.
    Este contexto não está vinculado a tags específicos, mas enriquece o entendimento geral da IA.
    """
    logger.info(f"Carregando contexto geral de {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()

        if not content:
            logger.warning(f"Arquivo de contexto {file_path} está vazio.")
            return []

        # Dividir o contexto em seções lógicas (por linhas em branco ou tópicos)
        sections = []
        current_section = []

        for line in content.split('\n'):
            line = line.strip()
            if line == "" and current_section:
                # Linha vazia indica fim de seção
                sections.append('\n'.join(current_section))
                current_section = []
            elif line:
                current_section.append(line)

        # Adicionar a última seção se existir
        if current_section:
            sections.append('\n'.join(current_section))

        # Se não conseguiu dividir em seções, usar o conteúdo completo
        if not sections:
            sections = [content]

        documents = []
        for i, section in enumerate(sections):
            doc = Document(
                page_content=section,
                metadata={
                    "source": file_path,
                    "section_id": i,
                    "data_type": "contexto_geral",
                    "topic": section.split(':')[0] if ':' in section else "contexto_geral"
                }
            )
            documents.append(doc)

        logger.info(f"Contexto geral dividido em {len(documents)} seções.")
        return documents

    except Exception as e:
        logger.error(f"Erro ao carregar contexto geral: {e}")
        return []

def load_and_chunk_csv(file_path: str, llm: Ollama, general_context: str = "") -> List[Document]:
    """
    **MODIFICADO: Carregamento e Geração de Contexto Híbrido**

    1. Lê o CSV.
    2. Para cada linha (TAG):
       a. Faz o parse do TAG em 3 componentes.
       b. Chama o LLM (Ollama) para gerar um "Contexto Descritivo" rico usando el modelo configurado.
    3. O 'page_content' (que será embedado) é esse Contexto Descritivo gerado.
    4. O 'nomeVariavel' literal e os componentes são guardados nos metadados.
    """
    logger.info(f"Carregando CSV de {file_path} para processamento híbrido...")
    try:
        # Definir tipos de dados para colunas problemáticas pode ajudar
        # dtype={'idElemento': str, 'idUnidade': str}
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Erro ao ler CSV: {e}")
        raise

    chunks = []
    total_rows = len(df)
    
    logger.info(f"Iniciando geração de contexto com LLM para {total_rows} TAGs. Isso pode demorar...")

    # Chunking a nível de linha com GERAÇÃO de contexto
    for index, row in df.iterrows():
        
        row_dict = {col: (None if pd.isna(row.get(col)) else row.get(col)) for col in df.columns}
        tag_literal = row_dict.get("nomeVariavel")

        # Pular linhas sem TAG literal, pois é a nossa chave
        if not tag_literal or tag_literal == 'ND':
            logger.warning(f"Pulando linha {index}: TAG ausente ou 'ND'.")
            continue
            
        # 1. Parsear o TAG
        components = parse_tag(tag_literal)
        if not components:
            logger.warning(f"Pulando linha {index}: TAG '{tag_literal}' não pôde ser parseado.")
            continue
            
        # 2. Gerar Contexto Descritivo com LLM
        # (Este é o passo que demora, pois chama a API do Ollama)
        logger.info(f"Processando [Linha {index+1}/{total_rows}]: {tag_literal}")
        
        descriptive_context = generate_descriptive_context(llm, components, row_dict, general_context)
        
        # 3. Criar Documento LangChain
        
        # O page_content é o contexto rico gerado pelo LLM (será embedado)
        page_content = descriptive_context

        # Metadados guardam as informações literais para filtragem e referência
        metadata = {
            "source": file_path,
            "row_id": index,
            "tag_literal": tag_literal, # (A) O TAG Literal
            "componente_1": components.get("componente_1"),
            "componente_2": components.get("componente_2"),
            "componente_3": components.get("componente_3"),
            "idVariavel": row_dict.get("idVariavel", ""),
            "nomeVariavel": row_dict.get("nomeVariavel", ""),
            "descricao_original": row_dict.get("descricao", ""), # Guardar a descrição original
            "grupamento": row_dict.get("grupamento", ""), # Nueva columna de grupamento
            "data_type": "dados_tabulares_tag"
        }
        
        doc = Document(
            page_content=page_content, # (B) O Contexto Descritivo (para embedding)
            metadata=metadata
        )
        chunks.append(doc)

    logger.info(f"CSV processado em {len(chunks)} chunks descritivos (Documentos).")
    return chunks

def initialize_tags_vector_store() -> ElasticsearchStore:
    """
    Inicializa el vector store para tags indexados localmente.

    Returns:
        Vector store configurado para tags
    """
    try:
        embeddings = create_embeddings_model()
        es_client = Elasticsearch(ELASTICSEARCH_URL)

        if es_client.indices.exists(index=INDEX_NAME_TAGS):
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
                logger.info(f"Vector store de tags creado exitosamente")
                return vector_store
            else:
                raise ValueError("No se pudieron procesar tags para indexar")

    except Exception as e:
        logger.error(f"Error inicializando vector store de tags: {e}")
        raise

def initialize_global_context_vector_store() -> ElasticsearchStore:
    """
    Inicializa el vector store para contexto global (documentación técnica).

    Returns:
        Vector store configurado para documentación global
    """
    try:
        embeddings = create_embeddings_model()
        es_client = Elasticsearch(ELASTICSEARCH_URL)

        if es_client.indices.exists(index=INDEX_NAME_GLOBAL_CONTEXT):
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
                logger.info(f"Vector store de contexto global creado exitosamente")
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

# Global variables - Arquitectura de dos índices
tags_vector_store = None
global_context_vector_store = None
llm_model = None

@app.on_event("startup")
async def startup_event():
    """Inicializar el sistema RAG con arquitectura de dos índices"""
    global tags_vector_store, global_context_vector_store, llm_model

    try:
        logger.info("Iniciando sistema RAG de Ingeniería de Reservatorios...")

        # Inicializar modelos
        llm_model = create_llm_model()
        logger.info("Modelo LLM (DeepSeek-R1) inicializado")

        # Inicializar índices separados
        tags_vector_store = initialize_tags_vector_store()
        logger.info("Vector store de tags inicializado")

        global_context_vector_store = initialize_global_context_vector_store()
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
        "modelo_llm": OLLAMA_MODEL
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
        events = generate_mock_events(start_date, end_date)

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
        if not all([tags_vector_store, global_context_vector_store, llm_model]):
            raise HTTPException(status_code=500, detail="Sistema RAG no inicializado completamente")

        logger.info(f"Procesando consulta inteligente: {request.question}")

        # =====================================================================
        # PASO 1: RECUPERAR EVENTOS (si la pregunta implica tiempo)
        # =====================================================================
        events_found = []
        question_lower = request.question.lower()

        # Detectar si la pregunta implica análisis temporal
        time_indicators = [
            "ayer", "hoy", "último", "última", "últimos", "últimas",
            "pasado", "reciente", "evento", "eventos", "sucede", "sucedió",
            "ocurrió", "ocurrir", "detectó", "detectado", "anomalía", "fallo"
        ]

        involves_time = any(indicator in question_lower for indicator in time_indicators)

        if involves_time:
            logger.info("Pregunta implica análisis temporal. Consultando eventos...")
            # Usar período de últimos 7 días por defecto
            end_date = datetime.now().date().isoformat()
            start_date = (datetime.now() - timedelta(days=7)).date().isoformat()

            events_response = await get_events(start_date, end_date)
            events_found = events_response.events[:5]  # Limitar a 5 eventos más relevantes
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

            # Crear retriever para tags con búsqueda híbrida
            tags_retriever = tags_vector_store.as_retriever(
                search_kwargs={"k": 3, "search_type": "hybrid"}
            )

            # Buscar información de cada tag
            for tag_id in tag_ids_from_events:
                try:
                    # Búsqueda exacta primero
                    exact_results = tags_vector_store.similarity_search_with_score(
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
                    related_results = tags_vector_store.similarity_search(
                        query=f"sensores relacionados a {tag_id}",
                        k=2
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

        global_retriever = global_context_vector_store.as_retriever(
            search_kwargs={"k": 4, "search_type": "hybrid"}
        )

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

        global_docs = global_retriever.get_relevant_documents(global_query)
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
    # Nota: O uvicorn.run() aqui é apenas para desenvolvimento. 
    # Em produção, use o entrypoint do Docker Compose.
    uvicorn.run(app, host="0.0.0.0", port=8000)