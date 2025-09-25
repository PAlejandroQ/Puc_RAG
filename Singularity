Bootstrap: docker
From: python:3.10-slim

%post
    # Update the system and install system dependencies
    apt-get update && apt-get install -y \
        curl \
        wget \
        && rm -rf /var/lib/apt/lists/*

    # Install Python dependencies
    pip install --no-cache-dir --upgrade pip
    pip install --no-cache-dir \
        langchain==0.3.27 \
        langchain-community==0.3.29 \
        pypdf==6.0.0 \
        elasticsearch \
        langchain-elasticsearch \
        ollama==0.5.4 \
        fastapi==0.116.1 \
        uvicorn[standard]==0.35.0

%environment
    # Environment variables for the RAG application
    export ELASTICSEARCH_URL=http://localhost:9200
    export OLLAMA_BASE_URL=http://localhost:11434
    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

%runscript
    # Default script when the container is run
    cd /app
    exec uvicorn main:app --host 0.0.0.0 --port 8000

%labels
    Author PUC RAG Team
    Version 1.0.0
    Description RAG Application container for document Q&A using Ollama and Elasticsearch

%help
    RAG Application Container
    =========================

    This Singularity container runs a Retrieval-Augmented Generation (RAG) system
    that provides a REST API for querying documents using FastAPI, LangChain, Ollama, and Elasticsearch.

    Usage:
    - Build: singularity build rag-app.sif Singularity
    - Run: singularity run --net --network-args="--portmap=8000:8000/tcp" rag-app.sif
    - Shell: singularity shell rag-app.sif

    Environment Variables:
    - ELASTICSEARCH_URL: URL for Elasticsearch service (default: http://localhost:9200)
    - OLLAMA_BASE_URL: URL for Ollama service (default: http://localhost:11434)

    Ports:
    - 8000: FastAPI application port

    The container expects to find the application code and document in /app directory.
