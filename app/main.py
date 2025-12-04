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
DOCUMENT_PATH_CSV = "app/csv/Tags_grupamento_nano.csv" # Fuente principal con columna grupamento
DOCUMENT_PATH_CSV_ADDITIONAL = "app/csv/tags_MRO3_nano.csv" # Datos adicionales opcionales
DOCUMENT_PATH_CONTEXT = "app/txt/context.txt" # Contexto general
INDEX_NAME_PDF = "codigo_civil_index"
INDEX_NAME_CSV = "tags_mro3_index_grupamento_nano2" # ATENÇÃO: Apague este índice no Elasticsearch para re-indexar
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")  # Modelo de Ollama a usar (para LLM y embeddings)
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
                # Carregar contexto geral primeiro para usar na geração de descrições
                general_context_text = ""
                if Path(DOCUMENT_PATH_CONTEXT).exists():
                    logger.info("Carregando contexto geral para enriquecer descrições...")
                    try:
                        with open(DOCUMENT_PATH_CONTEXT, 'r', encoding='utf-8') as f:
                            general_context_text = f.read().strip()
                        logger.info(f"Contexto geral carregado ({len(general_context_text)} caracteres)")
                    except Exception as e:
                        logger.warning(f"Erro ao carregar contexto geral: {e}")
                else:
                    logger.warning(f"Arquivo de contexto geral não encontrado: {DOCUMENT_PATH_CONTEXT}")

                # MODIFICADO: Criar um LLM *apenas* para o processo de ingestão
                logger.info("Criando instância do LLM (Ollama) para geração de contexto...")
                llm_ingestion = create_llm_model()

                # MODIFICADO: Passar o LLM e contexto geral para o loader
                chunks = load_and_chunk_csv(document_path, llm_ingestion, general_context_text)
                logger.info(f"Geração de contexto finalizada. {len(chunks)} documentos criados.")

                # Carregar contexto geral adicional como documentos separados para búsqueda
                if Path(DOCUMENT_PATH_CONTEXT).exists() and general_context_text:
                    logger.info("Adicionando contexto geral como documentos separados...")
                    general_context_docs = load_general_context(DOCUMENT_PATH_CONTEXT)
                    chunks.extend(general_context_docs)
                    logger.info(f"Contexto geral adicionado. Total de documentos: {len(chunks)}")
            
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