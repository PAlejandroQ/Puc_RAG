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
INDEX_NAME_CSV = "tags_pocos_mro_06_index_02"
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
    Loads and processes the CSV following best practices for RAG on tabular data.
    Each row is converted into a descriptive 'chunk' to improve embedding relevance.

    Args:
        file_path: Path to the CSV file

    Returns:
        List[Document]: List of processed documents
    """
    logger.info(f"Loading and processing CSV from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise

    chunks = []

    # Column descriptions for better understanding
    column_descriptions = {
        "Name": "Name of the tag or measurement point.",
        "Descriptor": "Human-readable description of the measurement point (the key for search).",
        "PointType": "Variable type (e.g., Int, Float, Digital).",
        "EngineeringUnits": "Variable unit (e.g., Binary, USD, kgf/cm² a, ºC, etc.).",
        "PointClass": "Class of the measurement point.",
        "Span": "Maximum value range.",
        "Zero": "Reference zero value.",
        "DigitalSetName": "Name of the digital set if applicable."
    }

    # Row-level chunking with context
    for index, row in df.iterrows():
        # Convert all NaN values to None to avoid serialization issues
        row_dict = {col: (None if pd.isna(row.get(col)) else row.get(col)) for col in df.columns}

        # Main information
        content_parts = []
        content_parts.append(f"The measurement point '{row_dict.get('Name', 'N/A')}' has the following description: '{row_dict.get('Descriptor', 'N/A')}'.")

        # Technical information
        if row_dict.get('PointType'):
            content_parts.append(f"It is of type '{row_dict['PointType']}'.")
        if row_dict.get('EngineeringUnits'):
            content_parts.append(f"Its units are '{row_dict['EngineeringUnits']}'.")
        if row_dict.get('PointClass'):
            content_parts.append(f"It belongs to the class '{row_dict['PointClass']}'.")

        # Additional technical information
        if row_dict.get('Span') is not None:
            content_parts.append(f"The maximum range is {row_dict['Span']}.")
        if row_dict.get('Zero') is not None:
            content_parts.append(f"The reference zero value is {row_dict['Zero']}.")
        if row_dict.get('DigitalSetName'):
            content_parts.append(f"It belongs to the digital set '{row_dict['DigitalSetName']}'.")

        content = " ".join(content_parts)

        # Create a LangChain Document object
        # Detailed metadata is added to facilitate future search and filtering
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
                "data_type": "tabular_data"  # Useful metadata to distinguish from PDFs
            }
        )
        chunks.append(doc)

    logger.info(f"CSV processed into {len(chunks)} descriptive chunks.")
    return chunks

def initialize_vector_store():
    """
    Initialize or load existing vector store with document embeddings using Elasticsearch.
    Supports modular loading of PDF or CSV based on the DATA_SOURCE variable.

    Returns:
        ElasticsearchStore: Configured vector store
    """
    try:
        # Determine paths and index names based on DATA_SOURCE
        if DATA_SOURCE == "PDF":
            current_index_name = INDEX_NAME_PDF
            document_path = DOCUMENT_PATH_PDF
            logger.info("Loading mode: PDF (Legal Document)")
        elif DATA_SOURCE == "CSV":
            current_index_name = INDEX_NAME_CSV
            document_path = DOCUMENT_PATH_CSV
            logger.info("Loading mode: CSV (Structured Data/Measurement Points)")
        else:
            raise ValueError(f"DATA_SOURCE '{DATA_SOURCE}' not supported. Must be 'PDF' or 'CSV'.")

        logger.info(f"Elasticsearch index: {current_index_name}")

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
                # Loading logic for PDF (Maintained)
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
                # Loading logic for CSV (Improvement: Descriptive Chunking)
                # Implementation of best practice: Row-level chunking with context
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

        # Create prompt template (in English)
        prompt_template = """You are an expert RAG assistant. Use only the provided context fragments to answer the question.
If the context refers to **tabular data/measurement points**, your answer must be precise and cite the value of the relevant columns (e.g., 'Name', 'Descriptor', 'PointType').
If the context does NOT have the information to answer, say you do not know. Do not make up answers.

Context:
{context}

Question: {question}

Concise answer (in English):"""

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
