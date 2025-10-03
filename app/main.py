"""
RAG Application using FastAPI, LangChain, Ollama, and ChromaDB
This application provides a REST API endpoint for querying documents using RAG.
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from elasticsearch import Elasticsearch
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Document Q&A", description="Query documents using RAG with Ollama and Elasticsearch")

# Configuration
DOCUMENT_PATH_PDF = "app/Codigo_Civil_split.pdf"
DOCUMENT_PATH_CSV = "app/csv/tags_pocos_mro_06_nano.csv"
INDEX_NAME_PDF = "codigo_civil_index"
INDEX_NAME_CSV = "tags_pocos_mro_06_index_01"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
DATA_SOURCE = os.getenv("DATA_SOURCE", "CSV").upper()  # 'PDF' o 'CSV'

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    question: str

class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str
    source_documents: List[Dict[str, Any]]

def create_embeddings_model() -> OllamaEmbeddings:
    """
    Create embeddings model instance.
    This function encapsulates the embeddings creation for easy future migration to HuggingFace.

    Returns:
        OllamaEmbeddings: Configured embeddings model
    """
    return OllamaEmbeddings(
        model="llama3",
        base_url=OLLAMA_BASE_URL
    )

def create_llm_model() -> Ollama:
    """
    Create LLM instance.
    This function encapsulates the LLM creation for easy future migration to HuggingFace.

    Returns:
        Ollama: Configured LLM instance
    """
    return Ollama(
        model="llama3",
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )

def load_and_chunk_csv(file_path: str) -> List[Document]:
    """
    Carga y procesa el CSV siguiendo las mejores prácticas para RAG en datos tabulares.
    Cada fila se convierte en un 'chunk' descriptivo para mejorar la relevancia del embedding.

    Args:
        file_path: Ruta al archivo CSV

    Returns:
        List[Document]: Lista de documentos procesados
    """
    logger.info(f"Cargando y procesando CSV desde {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error al leer el CSV: {e}")
        raise

    chunks = []

    # Definición de descripciones de columna para mejor comprensión
    column_descriptions = {
        "Name": "Nombre de la etiqueta o punto de medición.",
        "Descriptor": "Descripción entendible para humanos del punto de medición (la clave para la búsqueda).",
        "PointType": "Tipo de variable (ej: Int, Float, Digital).",
        "EngineeringUnits": "Unidad de la variable (ej: Binária, USD, kgf/cm² a, ºC, etc.).",
        "PointClass": "Clase del punto de medición.",
        "Span": "Rango máximo del valor.",
        "Zero": "Valor cero de referencia.",
        "DigitalSetName": "Nombre del conjunto digital si aplica."
    }

    # Chunking a Nivel de Fila con Contexto
    for index, row in df.iterrows():
        # Convertir todos los valores NaN a None para evitar problemas de serialización
        row_dict = {col: (None if pd.isna(row.get(col)) else row.get(col)) for col in df.columns}

        # Crear un texto descriptivo para el chunk (combinando la fila con sus encabezados)
        content_parts = []

        # Información principal
        content_parts.append(f"El punto de medición '{row_dict.get('Name', 'N/A')}' tiene la siguiente descripción: '{row_dict.get('Descriptor', 'N/A')}'.")

        # Información técnica
        if row_dict.get('PointType'):
            content_parts.append(f"Es de tipo '{row_dict['PointType']}'.")
        if row_dict.get('EngineeringUnits'):
            content_parts.append(f"Sus unidades son '{row_dict['EngineeringUnits']}'.")
        if row_dict.get('PointClass'):
            content_parts.append(f"Pertenece a la clase '{row_dict['PointClass']}'.")

        # Información adicional técnica
        if row_dict.get('Span') is not None:
            content_parts.append(f"El rango máximo es {row_dict['Span']}.")
        if row_dict.get('Zero') is not None:
            content_parts.append(f"El valor cero de referencia es {row_dict['Zero']}.")
        if row_dict.get('DigitalSetName'):
            content_parts.append(f"Pertenece al conjunto digital '{row_dict['DigitalSetName']}'.")

        content = " ".join(content_parts)

        # Crear un objeto Document de LangChain
        # Se añaden metadatos detallados para facilitar la búsqueda y filtrado futuro
        doc = Document(
            page_content=content,
            metadata={
                "source": file_path,
                "row_id": index,
                "WebId": row_dict.get("WebId", ""),
                "Id": row_dict.get("Id", ""),
                "Name": row_dict.get("Name", ""),
                "Path": row_dict.get("Path", ""),
                "Descriptor": row_dict.get("Descriptor", ""),
                "PointClass": row_dict.get("PointClass", ""),
                "PointType": row_dict.get("PointType", ""),
                "DigitalSetName": row_dict.get("DigitalSetName", ""),
                "EngineeringUnits": row_dict.get("EngineeringUnits", ""),
                "Span": row_dict.get("Span", None),
                "Zero": row_dict.get("Zero", None),
                "Step": row_dict.get("Step", None),
                "data_type": "tabular_data"  # Metadato útil para distinguir de PDFs
            }
        )
        chunks.append(doc)

    logger.info(f"CSV procesado en {len(chunks)} chunks descriptivos.")
    return chunks

def initialize_vector_store():
    """
    Initialize or load existing vector store with document embeddings using Elasticsearch.
    Soporta carga modular de PDF o CSV basada en la variable DATA_SOURCE.

    Returns:
        ElasticsearchStore: Configured vector store
    """
    try:
        # Determine paths and index names based on DATA_SOURCE
        if DATA_SOURCE == "PDF":
            current_index_name = INDEX_NAME_PDF
            document_path = DOCUMENT_PATH_PDF
            logger.info("Modo de carga: PDF (Documento legal)")
        elif DATA_SOURCE == "CSV":
            current_index_name = INDEX_NAME_CSV
            document_path = DOCUMENT_PATH_CSV
            logger.info("Modo de carga: CSV (Datos Estructurados/Puntos de Medición)")
        else:
            raise ValueError(f"DATA_SOURCE '{DATA_SOURCE}' no soportado. Debe ser 'PDF' o 'CSV'.")

        logger.info(f"Índice de Elasticsearch: {current_index_name}")

        # Initialize embeddings model
        embeddings = create_embeddings_model()
        es_client = Elasticsearch(ELASTICSEARCH_URL)

        # Check if index exists
        if es_client.indices.exists(index=current_index_name):
            logger.info("Found existing index, loading vector store...")
            vector_store = ElasticsearchStore(
                es_url=ELASTICSEARCH_URL,
                index_name=current_index_name,
                embedding=embeddings
            )
        else:
            logger.info("Index not found, creating new one...")

            if not Path(document_path).exists():
                raise FileNotFoundError(f"Document not found at {document_path}")

            if DATA_SOURCE == "PDF":
                # Lógica de carga para PDF (Mantenida)
                loader = PyPDFLoader(document_path)
                documents = loader.load()
                logger.info(f"Loaded {len(documents)} pages from PDF document")

                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_documents(documents)
                logger.info(f"Split PDF into {len(chunks)} chunks")

            elif DATA_SOURCE == "CSV":
                # Lógica de carga para CSV (Mejora: Chunking Descriptivo)
                # Implementación de la mejor práctica: Chunking a Nivel de Fila con Contexto
                chunks = load_and_chunk_csv(document_path)
                logger.info(f"Processed CSV into {len(chunks)} descriptive chunks")

            # Create vector store with embeddings
            vector_store = ElasticsearchStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                es_url=ELASTICSEARCH_URL,
                index_name=current_index_name
            )
            logger.info(f"Vector store created and populated for index {current_index_name}")

        return vector_store

    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        raise

# Global variables for vector store and QA chain
vector_store = None
qa_chain = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global vector_store, qa_chain

    try:
        # Initialize vector store
        vector_store = initialize_vector_store()

        # Create LLM
        llm = create_llm_model()

        # Create retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Create prompt template
        prompt_template = """Eres un asistente RAG experto. Utiliza únicamente los fragmentos de contexto proporcionados para responder a la pregunta.
        Si el contexto se refiere a **datos tabulares/puntos de medición**, tu respuesta debe ser precisa y citar el valor de las columnas relevantes (ej. 'Name', 'Descriptor', 'PointType').
        Si el contexto NO tiene la información para responder, di que no lo sabes. No inventes respuestas.

        Contexto:
        {context}

        Pregunta: {question}

        Respuesta concisa:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        logger.info("RAG system initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "RAG Document Q&A API", "status": "running"}

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query the document using RAG.

    Args:
        request: QueryRequest containing the question

    Returns:
        QueryResponse: Answer and source documents
    """
    try:
        if qa_chain is None:
            raise HTTPException(status_code=500, detail="RAG system not initialized")

        logger.info(f"Processing query: {request.question}")

        # Run the QA chain
        result = qa_chain({"query": request.question})

        # Format source documents
        source_documents = []
        for doc in result.get("source_documents", []):
            source_documents.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        return QueryResponse(
            answer=result["result"],
            source_documents=source_documents
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
