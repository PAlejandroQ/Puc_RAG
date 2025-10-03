#!/bin/bash

# =============================================================================
# Deployment Script for RAG System with Singularity
# =============================================================================
# This script deploys a RAG system equivalent to docker-compose.yml
# using Singularity for HPC environments.
#
# Services:
# - ollama: Language model
# - elasticsearch: Vector search engine
# - rag-app: FastAPI application
#
# Usage: ./deploy.sh [start|stop|status|logs]
# =============================================================================

set -e

# Output colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_FILE="${SCRIPT_DIR}/singularity_pids.txt"
LOGS_DIR="${SCRIPT_DIR}/logs"

# Create required directories
setup_directories() {
    echo -e "${BLUE}Creating required directories...${NC}"
    mkdir -p "${LOGS_DIR}"
    mkdir -p "${SCRIPT_DIR}/ollama_data"
    mkdir -p "${SCRIPT_DIR}/elasticsearch_data"
    mkdir -p "${SCRIPT_DIR}/app"
}

# Check if Singularity is installed
check_singularity() {
    if ! command -v singularity &> /dev/null; then
        echo -e "${RED}Error: Singularity is not installed.${NC}"
        echo "Install Singularity following the instructions at:"
        echo "https://docs.sylabs.io/guides/3.7/user-guide/installation.html"
        exit 1
    fi
    echo -e "${GREEN}✓ Singularity is installed${NC}"
}

# Wait for a service to be ready
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1

    echo -e "${YELLOW}Waiting for ${service_name} to be ready at ${url}...${NC}"

    while [ $attempt -le $max_attempts ]; do
        if curl -s "${url}" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ ${service_name} is ready${NC}"
            return 0
        fi

        echo -e "${YELLOW}Attempt ${attempt}/${max_attempts}: ${service_name} not ready yet...${NC}"
        sleep 10
        ((attempt++))
    done

    echo -e "${RED}Error: ${service_name} did not respond after ${max_attempts} attempts${NC}"
    return 1
}

# Start a service
start_service() {
    local service_name=$1
    local container_image=$2
    local singularity_cmd=$3
    local log_file="${LOGS_DIR}/${service_name}.log"

    echo -e "${BLUE}Starting ${service_name}...${NC}"

    # Run in background and capture PID
    nohup singularity run \
        --net \
        --network-args="--portmap=${singularity_cmd}" \
        --bind="${SCRIPT_DIR}/app:/app" \
        "${container_image}" > "${log_file}" 2>&1 &

    local pid=$!
    echo ${pid} >> "${PIDS_FILE}"

    echo -e "${GREEN}✓ ${service_name} started (PID: ${pid})${NC}"
    echo -e "${YELLOW}Logs: tail -f ${log_file}${NC}"
}

# Start all services
start_services() {
    echo -e "${BLUE}=== Starting RAG System ===${NC}"

    # Clear PIDs file
    > "${PIDS_FILE}"

    # Start Ollama
    echo -e "${BLUE}Starting Ollama...${NC}"
    nohup singularity run \
        --net \
        --network-args="--portmap=11434:11434/tcp" \
        --bind="${SCRIPT_DIR}/ollama_data:/root/.ollama" \
        docker://ollama/ollama:latest \
        serve > "${LOGS_DIR}/ollama.log" 2>&1 &

    echo $! >> "${PIDS_FILE}"
    echo -e "${GREEN}✓ Ollama started (PID: $!)${NC}"

    # Wait for Ollama to be ready
    sleep 5

    # Download llama3 model in Ollama
    echo -e "${BLUE}Downloading llama3 model in Ollama...${NC}"
    singularity exec docker://ollama/ollama:latest ollama pull llama3

    # Start Elasticsearch
    echo -e "${BLUE}Starting Elasticsearch...${NC}"
    nohup singularity run \
        --net \
        --network-args="--portmap=9200:9200/tcp" \
        --bind="${SCRIPT_DIR}/elasticsearch_data:/usr/share/elasticsearch/data" \
        --env="discovery.type=single-node" \
        --env="xpack.security.enabled=false" \
        --env="ES_JAVA_OPTS=-Xms512m -Xmx512m" \
        docker://docker.elastic.co/elasticsearch/elasticsearch:8.14.0 > "${LOGS_DIR}/elasticsearch.log" 2>&1 &

    echo $! >> "${PIDS_FILE}"
    echo -e "${GREEN}✓ Elasticsearch started (PID: $!)${NC}"

    # Wait for Elasticsearch to be ready
    if ! wait_for_service "Elasticsearch" "http://localhost:9200"; then
        echo -e "${RED}Could not connect to Elasticsearch. Checking logs...${NC}"
        tail -20 "${LOGS_DIR}/elasticsearch.log"
        exit 1
    fi

    # Start RAG application
    echo -e "${BLUE}Starting RAG application...${NC}"

    # Check if Singularity image exists
    if [ ! -f "rag-app.sif" ]; then
        echo -e "${YELLOW}Building Singularity image for RAG app...${NC}"
        singularity build rag-app.sif Singularity
    fi

    nohup singularity run \
        --net \
        --network-args="--portmap=8000:8000/tcp" \
        --bind="${SCRIPT_DIR}/app:/app" \
        rag-app.sif > "${LOGS_DIR}/rag-app.log" 2>&1 &

    echo $! >> "${PIDS_FILE}"
    echo -e "${GREEN}✓ RAG application started (PID: $!)${NC}"

    echo -e "${GREEN}=== RAG System successfully deployed ===${NC}"
    echo -e "${GREEN}API available at: http://localhost:8000${NC}"
    echo -e "${YELLOW}PIDs saved in: ${PIDS_FILE}${NC}"
    echo -e "${YELLOW}Logs in: ${LOGS_DIR}${NC}"
}

# Stop all services
stop_services() {
    echo -e "${BLUE}=== Stopping RAG System ===${NC}"

    if [ -f "${PIDS_FILE}" ]; then
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${YELLOW}Stopping process PID: ${pid}${NC}"
                kill "$pid"
                sleep 2
                if kill -0 "$pid" 2>/dev/null; then
                    echo -e "${YELLOW}Force killing process PID: ${pid}${NC}"
                    kill -9 "$pid"
                fi
                echo -e "${GREEN}✓ Process ${pid} stopped${NC}"
            else
                echo -e "${YELLOW}Process PID: ${pid} is not running${NC}"
            fi
        done < "${PIDS_FILE}"
        rm -f "${PIDS_FILE}"
        echo -e "${GREEN}=== All services stopped ===${NC}"
    else
        echo -e "${YELLOW}No PIDs file found. Services may not be running.${NC}"
    fi
}

# Show service status
show_status() {
    echo -e "${BLUE}=== RAG System Status ===${NC}"

    if [ -f "${PIDS_FILE}" ]; then
        echo -e "${GREEN}Active PIDs:${NC}"
        cat "${PIDS_FILE}"

        echo -e "\n${GREEN}Running processes:${NC}"
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                ps -p "$pid" -o pid,ppid,cmd,etime | tail -1
            fi
        done < "${PIDS_FILE}"
    else
        echo -e "${YELLOW}No registered services${NC}"
    fi

    # Check services by port
    echo -e "\n${GREEN}Port check:${NC}"
    if command -v netstat &> /dev/null; then
        netstat -tlnp 2>/dev/null | grep -E ':(11434|9200|8000)' || echo "netstat not available or ports not found"
    elif command -v ss &> /dev/null; then
        ss -tlnp | grep -E ':(11434|9200|8000)' || echo "ss not available or ports not found"
    else
        echo "No network tools available"
    fi
}

# Show service logs
show_logs() {
    echo -e "${BLUE}=== RAG System Logs ===${NC}"

    if [ -d "${LOGS_DIR}" ]; then
        for log_file in "${LOGS_DIR}"/*.log; do
            if [ -f "$log_file" ]; then
                service_name=$(basename "$log_file" .log)
                echo -e "\n${YELLOW}=== Logs for ${service_name} ===${NC}"
                tail -20 "$log_file"
            fi
        done
    else
        echo -e "${YELLOW}No logs directory found${NC}"
    fi
}

# Main function
main() {
    cd "${SCRIPT_DIR}"

    case "${1:-start}" in
        start)
            check_singularity
            setup_directories
            start_services
            ;;
        stop)
            stop_services
            ;;
        restart)
            stop_services
            sleep 3
            check_singularity
            setup_directories
            start_services
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        *)
            echo "Usage: $0 [start|stop|restart|status|logs]"
            echo ""
            echo "Commands:"
            echo "  start    - Start all services"
            echo "  stop     - Stop all services"
            echo "  restart  - Restart all services"
            echo "  status   - Show service status"
            echo "  logs     - Show service logs"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
