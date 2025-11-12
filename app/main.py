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

# Inicializar aplicação FastAPI
app = FastAPI(title="RAG Documentos Q&A", description="Consultar documentos usando RAG com Ollama e Elasticsearch")

# Configuração
DOCUMENT_PATH_PDF = "app/Codigo_Civil_split.pdf"
DOCUMENT_PATH_CSV = "app/csv/tags_MRO3_nano.csv"
INDEX_NAME_PDF = "codigo_civil_index"
INDEX_NAME_CSV = "tags_mro3_index_nano"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ELASTICSEARCH_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
DATA_SOURCE = os.getenv("DATA_SOURCE", "CSV").upper()  # 'PDF' ou 'CSV'

class QueryRequest(BaseModel):
    """Modelo de requisição para o endpoint de consulta"""
    question: str

class QueryResponse(BaseModel):
    """Modelo de resposta para o endpoint de consulta"""
    answer: str
    source_documents: List[Dict[str, Any]]

def create_embeddings_model() -> OllamaEmbeddings:
    """
    Criar instância do modelo de embeddings.
    Esta função encapsula a criação de embeddings para facilitar futura migração para HuggingFace.

    Returns:
        OllamaEmbeddings: Modelo de embeddings configurado
    """
    return OllamaEmbeddings(
        model="llama3",
        base_url=OLLAMA_BASE_URL
    )

def create_llm_model() -> Ollama:
    """
    Criar instância do LLM.
    Esta função encapsula a criação do LLM para facilitar futura migração para HuggingFace.

    Returns:
        Ollama: Instância do LLM configurado
    """
    return Ollama(
        model="llama3",
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )

def load_and_chunk_csv(file_path: str) -> List[Document]:
    """
    Carrega e processa o CSV seguindo as melhores práticas para RAG em dados tabulares.
    Cada linha é convertida em um 'chunk' descritivo para melhorar a relevância do embedding.

    Args:
        file_path: Caminho para o arquivo CSV

    Returns:
        List[Document]: Lista de documentos processados
    """
    logger.info(f"Carregando e processando CSV de {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Erro ao ler CSV: {e}")
        raise

    chunks = []

    # Descrições das colunas para melhor compreensão
    column_descriptions = {
        "nomeVariavel": "Nome da variável ou ponto de medição.",
        "descricao": "Descrição legível para humanos do ponto de medição (a chave para busca).",
        "idElemento": "ID do elemento/poço.",
        "nomeElemento": "Nome do elemento/poço.",
        "idUnidade": "ID da unidade de medida.",
        "siglaUnidade": "Sigla da unidade de medida.",
        "tag": "Tag ou identificador da variável."
    }

    # Chunking a nível de linha com contexto
    for index, row in df.iterrows():
        # Converter todos os valores NaN para None para evitar problemas de serialização
        row_dict = {col: (None if pd.isna(row.get(col)) else row.get(col)) for col in df.columns}

        # Informações principais
        content_parts = []
        content_parts.append(f"A variável '{row_dict.get('nomeVariavel', 'N/A')}' tem a seguinte descrição: '{row_dict.get('descricao', 'N/A')}'.")

        # Informações técnicas adicionais
        # if row_dict.get('nomeElemento'):
        #     content_parts.append(f"Pertence ao elemento '{row_dict['nomeElemento']}'.")
        # if row_dict.get('siglaUnidade'):
        #     content_parts.append(f"Sua unidade é '{row_dict['siglaUnidade']}'.")
        # if row_dict.get('tag') and row_dict['tag'] != 'ND':
        #     content_parts.append(f"Seu tag é '{row_dict['tag']}'.")

        content = " ".join(content_parts)

        # Criar um objeto Document do LangChain
        # Metadados detalhados são adicionados para facilitar futuras buscas e filtros
        doc = Document(
            page_content=content,
            metadata={
                "source": file_path,
                "row_id": index,
                "idVariavel": row_dict.get("idVariavel", ""),
                "nomeVariavel": row_dict.get("nomeVariavel", ""),
                "descricao": row_dict.get("descricao", ""),
                "idElemento": row_dict.get("idElemento", ""),
                "nomeElemento": row_dict.get("nomeElemento", ""),
                "idUnidade": row_dict.get("idUnidade", ""),
                "siglaUnidade": row_dict.get("siglaUnidade", ""),
                "tag": row_dict.get("tag", ""),
                "data_type": "dados_tabulares"  # Metadado útil para distinguir de PDFs
            }
        )
        chunks.append(doc)

    logger.info(f"CSV processado em {len(chunks)} chunks descritivos.")
    return chunks

def initialize_vector_store():
    """
    Inicializa ou carrega vector store existente com embeddings de documentos usando Elasticsearch.
    Suporta carregamento modular de PDF ou CSV baseado na variável DATA_SOURCE.

    Returns:
        ElasticsearchStore: Vector store configurado
    """
    try:
        # Determinar caminhos e nomes de índices baseado no DATA_SOURCE
        if DATA_SOURCE == "PDF":
            current_index_name = INDEX_NAME_PDF
            document_path = DOCUMENT_PATH_PDF
            logger.info("Modo de carregamento: PDF (Documento Legal)")
        elif DATA_SOURCE == "CSV":
            current_index_name = INDEX_NAME_CSV
            document_path = DOCUMENT_PATH_CSV
            logger.info("Modo de carregamento: CSV (Dados Estruturados/Variáveis de Medição)")
        else:
            raise ValueError(f"DATA_SOURCE '{DATA_SOURCE}' não suportado. Deve ser 'PDF' ou 'CSV'.")

        logger.info(f"Índice do Elasticsearch: {current_index_name}")

        # Inicializar modelo de embeddings
        embeddings = create_embeddings_model()
        es_client = Elasticsearch(ELASTICSEARCH_URL)

        # Verificar se o índice existe
        if es_client.indices.exists(index=current_index_name):
            logger.info("Índice existente encontrado, carregando vector store...")
            vector_store = ElasticsearchStore(
                es_url=ELASTICSEARCH_URL,
                index_name=current_index_name,
                embedding=embeddings
            )
        else:
            logger.info("Índice não encontrado, criando novo...")

            if not Path(document_path).exists():
                raise FileNotFoundError(f"Documento não encontrado em {document_path}")

            if DATA_SOURCE == "PDF":
                # Lógica de carregamento para PDF (Mantida)
                loader = PyPDFLoader(document_path)
                documents = loader.load()
                logger.info(f"Carregadas {len(documents)} páginas do documento PDF")

                # Dividir documentos em chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_documents(documents)
                logger.info(f"PDF dividido em {len(chunks)} chunks")

            elif DATA_SOURCE == "CSV":
                # Lógica de carregamento para CSV (Melhoria: Chunking Descritivo)
                # Implementação da melhor prática: Chunking a nível de linha com contexto
                chunks = load_and_chunk_csv(document_path)
                logger.info(f"CSV processado em {len(chunks)} chunks descritivos")

            # Criar vector store com embeddings
            vector_store = ElasticsearchStore.from_documents(
                documents=chunks,
                embedding=embeddings,
                es_url=ELASTICSEARCH_URL,
                index_name=current_index_name
            )
            logger.info(f"Vector store criado e populado para o índice {current_index_name}")

        return vector_store

    except Exception as e:
        logger.error(f"Erro ao inicializar vector store: {str(e)}")
        raise

# Global variables for vector store and QA chain
vector_store = None
qa_chain = None

@app.on_event("startup")
async def startup_event():
    """Inicializar o sistema RAG na inicialização"""
    global vector_store, qa_chain

    try:
        # Inicializar vector store
        vector_store = initialize_vector_store()

        # Criar LLM
        llm = create_llm_model()

        # Criar retriever
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Criar template de prompt (em português)
        prompt_template = """Você é um assistente RAG especialista. Use apenas os fragmentos de contexto fornecidos para responder à pergunta.
Se o contexto se refere a **dados tabulares/variáveis de medição**, sua resposta deve ser precisa e citar o valor das colunas relevantes (ex.: 'nomeVariavel', 'descricao', 'siglaUnidade').
Se o contexto NÃO tiver a informação para responder, diga que não sabe. Não invente respostas.

Contexto:
{context}

Pergunta: {question}

Resposta concisa (em português):"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Criar cadeia QA
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        logger.info("Sistema RAG inicializado com sucesso")

    except Exception as e:
        logger.error(f"Falha ao inicializar sistema RAG: {str(e)}")
        raise

@app.get("/")
async def root():
    """Endpoint de verificação de saúde"""
    return {"message": "API RAG Documentos Q&A", "status": "executando"}

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Consultar o documento usando RAG.

    Args:
        request: QueryRequest contendo a pergunta

    Returns:
        QueryResponse: Resposta e documentos fonte
    """
    try:
        if qa_chain is None:
            raise HTTPException(status_code=500, detail="Sistema RAG não inicializado")

        logger.info(f"Processando consulta: {request.question}")

        # Executar a cadeia QA
        result = qa_chain({"query": request.question})

        # Formatar documentos fonte
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
        logger.error(f"Erro ao processar consulta: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar consulta: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
