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
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import chromadb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG Document Q&A", description="Query documents using RAG with Ollama and ChromaDB")

# Configuration
DOCUMENT_PATH = "app/document.pdf"
COLLECTION_NAME = "document_collection2"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", 8001)

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

def initialize_vector_store():
    """
    Initialize or load existing vector store with document embeddings.

    This function:
    1. Checks if collection exists in ChromaDB
    2. If not, loads the PDF, splits it into chunks, creates embeddings, and stores them
    3. If exists, simply connects to the existing collection

    Returns:
        Chroma: Configured vector store
    """
    try:
        logger.info("Initializing vector store...")

        # Initialize embeddings model
        embeddings = create_embeddings_model()

        # Connect to ChromaDB
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

        # Check if collection exists
        try:
            collection = chroma_client.get_collection(COLLECTION_NAME)
            logger.info("Found existing collection, loading vector store...")
            vector_store = Chroma(
                client=chroma_client,
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings
            )
        except Exception:
            logger.info("Collection not found, creating new one...")

            # Load and process document
            if not Path(DOCUMENT_PATH).exists():
                raise FileNotFoundError(f"Document not found at {DOCUMENT_PATH}")

            loader = PyPDFLoader(DOCUMENT_PATH)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from document")

            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(chunks)} chunks")

            # Create vector store with embeddings
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                client=chroma_client,
                collection_name=COLLECTION_NAME
            )
            logger.info("Vector store created and populated")

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
        prompt_template = """Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}

        Answer:"""

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
