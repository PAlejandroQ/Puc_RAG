"""
Document Processing Utilities for RAG System.

Handles PDF loading, text splitting, and document chunking for global context.
"""

import logging
from pathlib import Path
from typing import List

import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.config import (
    DOCUMENT_PATH_PDF_1,
    DOCUMENT_PATH_PDF_2,
    PDF_CHUNK_SIZE,
    PDF_CHUNK_OVERLAP,
    PDF_CHUNK_SEPARATORS
)

logger = logging.getLogger(__name__)


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
                    chunk_size=PDF_CHUNK_SIZE,  # Chunks más grandes para contexto técnico
                    chunk_overlap=PDF_CHUNK_OVERLAP,
                    separators=PDF_CHUNK_SEPARATORS
                )

                chunks = text_splitter.split_documents(pdf_docs)

                # Agregar metadata específica
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "source": str(pdf_path),
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


def load_general_context_from_file(file_path: str) -> List[Document]:
    """
    Carga el contexto general del archivo de texto.
    Este contexto no está vinculado a tags específicos, pero enriquece el entendimiento general de la IA.
    """
    logger.info(f"Cargando contexto general de {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()

        if not content:
            logger.warning(f"Archivo de contexto {file_path} está vacío.")
            return []

        # Dividir el contexto en secciones lógicas (por líneas en blanco o tópicos)
        sections = []
        current_section = []

        for line in content.split('\n'):
            line = line.strip()
            if line == "" and current_section:
                # Línea vacía indica fin de sección
                sections.append('\n'.join(current_section))
                current_section = []
            elif line:
                current_section.append(line)

        # Agregar la última sección si existe
        if current_section:
            sections.append('\n'.join(current_section))

        # Si no consiguió dividir en secciones, usar el contenido completo
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

        logger.info(f"Contexto general dividido en {len(documents)} secciones.")
        return documents

    except Exception as e:
        logger.error(f"Error cargando contexto general: {e}")
        return []


def create_documents_from_csv(csv_path: str, llm, general_context: str = "") -> List[Document]:
    """
    **MODIFICADO: Carga y Generación de Contexto Híbrido**

    1. Lee el CSV.
    2. Para cada línea (TAG):
       a. Hace el parse del TAG en 3 componentes.
       b. Llama al LLM para generar un "Contexto Descritivo" rico usando el modelo configurado.
    3. El 'page_content' (que será embedado) es ese Contexto Descritivo generado.
    4. El 'nomeVariavel' literal y los componentes son guardados en los metadados.
    """
    from app.utils.tag_processing import parse_tag, generate_descriptive_context

    logger.info(f"Cargando CSV de {csv_path} para procesamiento híbrido...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Error leyendo CSV: {e}")
        raise

    chunks = []
    total_rows = len(df)

    logger.info(f"Iniciando generación de contexto con LLM para {total_rows} TAGs. Esto puede demorar...")

    # Chunking a nivel de línea con GENERACIÓN de contexto
    for index, row in df.iterrows():

        row_dict = {col: (None if pd.isna(row.get(col)) else row.get(col)) for col in df.columns}
        tag_literal = row_dict.get("nomeVariavel")

        # Pular linhas sem TAG literal, pois é a nossa chave
        if not tag_literal or tag_literal == 'ND':
            logger.warning(f"Saltando línea {index}: TAG ausente o 'ND'.")
            continue

        # 1. Parsear el TAG
        components = parse_tag(tag_literal)
        if not components:
            logger.warning(f"Saltando línea {index}: TAG '{tag_literal}' no pudo ser parseado.")
            continue

        # 2. Generar Contexto Descritivo con LLM
        # (Este es el paso que demora, pues llama a la API del Ollama)
        logger.info(f"Procesando [Línea {index+1}/{total_rows}]: {tag_literal}")

        descriptive_context = generate_descriptive_context(llm, components, row_dict, general_context)

        # 3. Crear Documento LangChain

        # El page_content es el contexto rico generado por el LLM (será embedado)
        page_content = descriptive_context

        # Metadatos guardan las informaciones literales para filtrado y referencia
        metadata = {
            "source": csv_path,
            "row_id": index,
            "tag_literal": tag_literal, # (A) El TAG Literal
            "componente_1": components.componente_1,
            "componente_2": components.componente_2,
            "componente_3": components.componente_3,
            "idVariavel": row_dict.get("idVariavel", ""),
            "nomeVariavel": row_dict.get("nomeVariavel", ""),
            "descricao_original": row_dict.get("descricao", ""), # Guardar la descripción original
            "grupamento": row_dict.get("grupamento", ""), # Nueva columna de grupamento
            "data_type": "dados_tabulares_tag"
        }

        doc = Document(
            page_content=page_content, # (B) El Contexto Descritivo (para embedding)
            metadata=metadata
        )
        chunks.append(doc)

    logger.info(f"CSV procesado en {len(chunks)} chunks descriptivos (Documentos).")
    return chunks