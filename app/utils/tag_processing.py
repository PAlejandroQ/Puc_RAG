"""
Tag Processing Utilities for RAG System.

Handles tag parsing, inference, and descriptive context generation.
"""

import re
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from app.config import DOCUMENT_PATH_TAGS_CSV
from app.services.embeddings_service import embeddings_service
from app.models import TagComponents

logger = logging.getLogger(__name__)


def parse_tag(tag_string: str) -> Optional[TagComponents]:
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
        return TagComponents(
            componente_1=match.group(1),
            componente_2=match.group(2),
            componente_3=match.group(3)
        )
    else:
        # Tentar um padrão para tags como POCO_MRO_003_SCM_IWIS_MA_06_STATUS
        # Onde componente 1 é POCO_MRO_XXX (XXX sendo números) e componente 2 é o resto
        match_poco = re.match(r"^(POCO_MRO_\d+)_(.+)$", tag_string)
        if match_poco:
            return TagComponents(
                componente_1=match_poco.group(1),  # POCO_MRO_003
                componente_2=match_poco.group(2),  # SCM_IWIS_MA_06_STATUS
                componente_3=None  # No hay componente 3 en este formato
            )
        else:
            # Tentar um padrão mais simples se o primeiro falhar (ex: TAGs com apenas 2 componentes)
            match_simple = re.match(r"^(\S+)\s+(.+)$", tag_string)
            if match_simple:
                 return TagComponents(
                    componente_1=match_simple.group(1),
                    componente_2=None, # Não corresponde ao padrão de 3
                    componente_3=match_simple.group(2)
                )
    logger.warning(f"TAG '{tag_string}' não correspondeu a nenhum padrão de parse.")
    return None


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


def generate_descriptive_context(llm, components: TagComponents, row: Dict[str, Any], general_context: str = "") -> str:
    """
    Usa el LLM configurado para generar un "Contexto Descritivo" rico para el embedding.
    Combina los componentes del TAG con los datos originales del CSV y usa el contexto general para enriquecer.
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
        "componente_1": components.componente_1 if components else "N/A",
        "componente_2": components.componente_2 if components else "N/A",
        "componente_3": components.componente_3 if components else "N/A"
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