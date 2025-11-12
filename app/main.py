"""
RAG Application using FastAPI, LangChain, Ollama, and Elasticsearch
Implementa um RAG Híbrido (Semântico + Literal) para TAGs de engenharia.
"""

import os
import logging
import re # Adicionado para Regex
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
app = FastAPI(title="RAG Híbrido de TAGs Q&A", description="Consultar TAGs de engenharia usando RAG Híbrido com Ollama e Elasticsearch")

# Configuração
DOCUMENT_PATH_PDF = "app/Codigo_Civil_split.pdf"
DOCUMENT_PATH_CSV = "app/csv/tags_MRO3_nano.csv" # Certifique-se que este é o caminho correto
INDEX_NAME_PDF = "codigo_civil_index"
INDEX_NAME_CSV = "tags_mro3_index_nano2" # ATENÇÃO: Apague este índice no Elasticsearch para re-indexar
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

# =============================================================================
# NOVAS FUNÇÕES HELPER (Parse e Geração de Contexto)
# =============================================================================

def parse_tag(tag_string: str) -> Optional[Dict[str, str]]:
    """
    Divide um TAG literal nos 3 componentes identificados usando regex.
    Componente 1: (Tipo) Sem espaço
    Componente 2: (ID Poço) Sem espaço
    Componente 3: (Descrição/Local) Restante da string

    Exemplo: "PI 7-MRO-3-RJS PDG Coluna acima do Packer"
    Retorna: {
        'componente_1': 'PI',
        'componente_2': '7-MRO-3-RJS',
        'componente_3': 'PDG Coluna acima do Packer'
    }
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

def generate_descriptive_context(llm: Ollama, components: Dict[str, str], row: Dict[str, Any]) -> str:
    """
    Usa o LLM (llama3) para gerar um "Contexto Descritivo" rico para o embedding.
    Combina os componentes do TAG com os dados originais do CSV.
    """
    
    # Template do prompt para o LLM de geração
    generation_template = """Você é um especialista em engenharia de petróleo e instrumentação (Oil & Gas).
Sua tarefa é gerar uma descrição técnica unificada e rica em contexto para um TAG de medição, com base nas informações fornecidas.
Esta descrição será usada para busca semântica (embedding), então deve ser completa.

**Informações Fornecidas:**
- **TAG Literal:** {tag_literal}
- **Descrição Original:** {descricao}
- **Componente 1 (Tipo):** {componente_1}
- **Componente 2 (ID Poço/Sistema):** {componente_2}
- **Componente 3 (Medição/Local):** {componente_3}
- **Nome do Elemento (Poço):** {nomeElemento}
- **Unidade de Medida:** {siglaUnidade}

**Instrução:**
Gere um parágrafo conciso (em português) que descreva este ponto de medição. Combine todas as informações de forma lógica.
Exemplo de Resposta: "Este é um Ponto de Instrumentação (PI) do tipo Medidor Permanente de Fundo de Poço (PDG) localizado na Coluna Inferior, referente ao poço 7-MRO-3-RJS. Ele mede a Pressão (P) em bar."

**Contexto Descritivo Unificado (para embedding):**
"""
    
    prompt = PromptTemplate(
        template=generation_template,
        input_variables=[
            "tag_literal", "descricao", "componente_1", "componente_2", 
            "componente_3", "nomeElemento", "siglaUnidade"
        ]
    )

    # Preencher com os dados da linha, tratando valores ausentes
    input_data = {
        "tag_literal": row.get("nomeVariavel", "N/A"),
        "descricao": row.get("descricao", "N/A"),
        "componente_1": components.get("componente_1", "N/A"),
        "componente_2": components.get("componente_2", "N/A"),
        "componente_3": components.get("componente_3", "N/A"),
        "nomeElemento": row.get("nomeElemento", "N/A"),
        "siglaUnidade": row.get("siglaUnidade", "N/A")
    }

    try:
        # Criar a cadeia e invocar o LLM
        chain = prompt | llm
        descriptive_context = chain.invoke(input_data)
        return descriptive_context.strip()
    except Exception as e:
        logger.error(f"Erro ao gerar contexto para o TAG '{row.get('nomeVariavel')}': {e}")
        # Retornar uma descrição de fallback caso o LLM falhe
        return f"TAG: {row.get('nomeVariavel', 'N/A')}. Descrição: {row.get('descricao', 'N/A')}. Elemento: {row.get('nomeElemento', 'N/A')}."


# =============================================================================
# FUNÇÕES PRINCIPAIS (Modificadas)
# =============================================================================

def create_embeddings_model() -> OllamaEmbeddings:
    """Criar instância do modelo de embeddings."""
    return OllamaEmbeddings(
        model="llama3", # Usar llama3 para embeddings também
        base_url=OLLAMA_BASE_URL
    )

def create_llm_model() -> Ollama:
    """Criar instância do LLM."""
    return Ollama(
        model="llama3",
        base_url=OLLAMA_BASE_URL,
        temperature=0.1
    )

def load_and_chunk_csv(file_path: str, llm: Ollama) -> List[Document]:
    """
    **MODIFICADO: Carregamento e Geração de Contexto Híbrido**
    
    1. Lê o CSV.
    2. Para cada linha (TAG):
       a. Faz o parse do TAG em 3 componentes.
       b. Chama o LLM (Ollama) para gerar um "Contexto Descritivo" rico.
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
        
        descriptive_context = generate_descriptive_context(llm, components, row_dict)
        
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
            "idElemento": row_dict.get("idElemento", ""),
            "nomeElemento": row_dict.get("nomeElemento", ""),
            "siglaUnidade": row_dict.get("siglaUnidade", ""),
            "data_type": "dados_tabulares_tag"
        }
        
        doc = Document(
            page_content=page_content, # (B) O Contexto Descritivo (para embedding)
            metadata=metadata
        )
        chunks.append(doc)

    logger.info(f"CSV processado em {len(chunks)} chunks descritivos (Documentos).")
    return chunks

def initialize_vector_store():
    """
    **MODIFICADO: Inicializa o vector store.**
    
    - Se o índice não existe, chama o `create_llm_model` e passa o LLM
      para `load_and_chunk_csv` para gerar os contextos descritivos.
    - Se o índice já existe, apenas se conecta a ele.
    """
    try:
        if DATA_SOURCE == "CSV":
            current_index_name = INDEX_NAME_CSV
            document_path = DOCUMENT_PATH_CSV
            logger.info("Modo de carregamento: CSV (Dados Estruturados/TAGs)")
        elif DATA_SOURCE == "PDF":
            current_index_name = INDEX_NAME_PDF
            document_path = DOCUMENT_PATH_PDF
            logger.info("Modo de carregamento: PDF (Documento Legal)")
        else:
            raise ValueError(f"DATA_SOURCE '{DATA_SOURCE}' não suportado.")

        logger.info(f"Índice do Elasticsearch: {current_index_name}")

        embeddings = create_embeddings_model()
        es_client = Elasticsearch(ELASTICSEARCH_URL)

        if es_client.indices.exists(index=current_index_name):
            logger.info("Índice existente encontrado. Carregando vector store...")
            logger.info("NOTA: Para re-indexar com a nova lógica (geração de contexto), apague este índice no Elasticsearch.")
            vector_store = ElasticsearchStore(
                es_url=ELASTICSEARCH_URL,
                index_name=current_index_name,
                embedding=embeddings
            )
        else:
            logger.info(f"Índice '{current_index_name}' não encontrado. Criando novo...")
            logger.info("Este processo envolve a geração de contexto por LLM e pode demorar.")

            if not Path(document_path).exists():
                raise FileNotFoundError(f"Documento não encontrado em {document_path}")

            if DATA_SOURCE == "CSV":
                # MODIFICADO: Criar um LLM *apenas* para o processo de ingestão
                logger.info("Criando instância do LLM (Ollama) para geração de contexto...")
                llm_ingestion = create_llm_model()
                
                # MODIFICADO: Passar o LLM para o loader
                chunks = load_and_chunk_csv(document_path, llm_ingestion)
                logger.info(f"Geração de contexto finalizada. {len(chunks)} documentos criados.")
            
            # elif DATA_SOURCE == "PDF":
            #     # Lógica de carregamento para PDF (Mantida)
            #     loader = PyPDFLoader(document_path)
            #     documents = loader.load()
            #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            #     chunks = text_splitter.split_documents(documents)
            #     logger.info(f"PDF dividido em {len(chunks)} chunks")

            # Criar vector store com embeddings
            logger.info(f"Iniciando indexação de {len(chunks)} documentos no Elasticsearch...")
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

# Global variables
vector_store = None
qa_chain = None

@app.on_event("startup")
async def startup_event():
    """Inicializar o sistema RAG na inicialização"""
    global vector_store, qa_chain

    try:
        vector_store = initialize_vector_store()
        llm = create_llm_model()

        # =====================================================================
        # MUDANÇA PRINCIPAL: HABILITANDO A BUSCA HÍBRIDA
        # =====================================================================
        # Ao invés de usar o retriever padrão (só vetorial),
        # usamos search_type="hybrid".
        # O Elasticsearch agora combinará a busca vetorial (semântica)
        # com a busca de keyword (BM25) no 'page_content' (Contexto Descritivo).
        
        retriever = vector_store.as_retriever(
            search_kwargs={
                    "k": 5,
                    "search_type": "hybrid"
                } # Aumentar para 5 para dar mais chance à hibridização
        )
        
        logger.info("Retriever configurado para BUSCA HÍBRIDA (Semântica + Keyword).")

        # Template de prompt (em português)
        # Este prompt funciona bem para o novo contexto
        prompt_template = """Você é um assistente RAG especialista em engenharia de petróleo. Use apenas os fragmentos de contexto fornecidos (que são descrições de TAGs) para responder à pergunta.

Seja preciso e cite o TAG literal (ex: "PI 7-MRO-3-RJS...") e outras informações relevantes dos metadados (como 'nomeElemento' ou 'siglaUnidade') se ajudarem a responder.
Se o contexto NÃO tiver a informação para responder, diga que não sabe.

Contexto:
{context}

Pergunta: {question}

Resposta concisa (em português):"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

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
    return {"message": "API RAG Híbrido de TAGs", "status": "executando"}

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Consultar o documento usando RAG Híbrido."""
    try:
        if qa_chain is None:
            raise HTTPException(status_code=500, detail="Sistema RAG não inicializado")

        logger.info(f"Processando consulta HÍBRIDA: {request.question}")
        
        # O retriever híbrido buscará tanto pela semântica da pergunta
        # (ex: "vazão de óleo na válvula superior")
        # quanto pelas keywords (ex: "Qo", "ICV", "sup")
        # contra o 'page_content' (Contexto Descritivo) que geramos.
        
        result = qa_chain({"query": request.question})

        source_documents = []
        for doc in result.get("source_documents", []):
            source_documents.append({
                "content_gerado": doc.page_content, # O contexto que o LLM gerou
                "metadata": doc.metadata # Onde está o tag_literal e outros dados
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
    # Nota: O uvicorn.run() aqui é apenas para desenvolvimento. 
    # Em produção, use o entrypoint do Docker Compose.
    uvicorn.run(app, host="0.0.0.0", port=8000)