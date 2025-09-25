#!/bin/bash

# =============================================================================
# Script de Despliegue para Sistema RAG con Singularity
# =============================================================================
# Este script despliega un sistema RAG equivalente al docker-compose.yml
# usando Singularity para entornos HPC.
#
# Servicios:
# - ollama: Modelo de lenguaje
# - elasticsearch: Motor de búsqueda vectorial
# - rag-app: Aplicación FastAPI
#
# Uso: ./deploy.sh [start|stop|status|logs]
# =============================================================================

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuración
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_FILE="${SCRIPT_DIR}/singularity_pids.txt"
LOGS_DIR="${SCRIPT_DIR}/logs"

# Crear directorios necesarios
setup_directories() {
    echo -e "${BLUE}Creando directorios necesarios...${NC}"
    mkdir -p "${LOGS_DIR}"
    mkdir -p "${SCRIPT_DIR}/ollama_data"
    mkdir -p "${SCRIPT_DIR}/elasticsearch_data"
    mkdir -p "${SCRIPT_DIR}/app"
}

# Verificar si Singularity está instalado
check_singularity() {
    if ! command -v singularity &> /dev/null; then
        echo -e "${RED}Error: Singularity no está instalado.${NC}"
        echo "Instale Singularity siguiendo las instrucciones en:"
        echo "https://docs.sylabs.io/guides/3.7/user-guide/installation.html"
        exit 1
    fi
    echo -e "${GREEN}✓ Singularity está instalado${NC}"
}

# Función para esperar a que un servicio esté listo
wait_for_service() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1

    echo -e "${YELLOW}Esperando a que ${service_name} esté listo en ${url}...${NC}"

    while [ $attempt -le $max_attempts ]; do
        if curl -s "${url}" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ ${service_name} está listo${NC}"
            return 0
        fi

        echo -e "${YELLOW}Intento ${attempt}/${max_attempts}: ${service_name} no está listo aún...${NC}"
        sleep 10
        ((attempt++))
    done

    echo -e "${RED}Error: ${service_name} no respondió después de ${max_attempts} intentos${NC}"
    return 1
}

# Función para iniciar un servicio
start_service() {
    local service_name=$1
    local container_image=$2
    local singularity_cmd=$3
    local log_file="${LOGS_DIR}/${service_name}.log"

    echo -e "${BLUE}Iniciando ${service_name}...${NC}"

    # Ejecutar en segundo plano y capturar PID
    nohup singularity run \
        --net \
        --network-args="--portmap=${singularity_cmd}" \
        --bind="${SCRIPT_DIR}/app:/app" \
        "${container_image}" > "${log_file}" 2>&1 &

    local pid=$!
    echo ${pid} >> "${PIDS_FILE}"

    echo -e "${GREEN}✓ ${service_name} iniciado (PID: ${pid})${NC}"
    echo -e "${YELLOW}Logs: tail -f ${log_file}${NC}"
}

# Iniciar servicios
start_services() {
    echo -e "${BLUE}=== Iniciando Sistema RAG ===${NC}"

    # Limpiar archivo de PIDs
    > "${PIDS_FILE}"

    # Iniciar Ollama
    echo -e "${BLUE}Iniciando Ollama...${NC}"
    nohup singularity run \
        --net \
        --network-args="--portmap=11434:11434/tcp" \
        --bind="${SCRIPT_DIR}/ollama_data:/root/.ollama" \
        docker://ollama/ollama:latest \
        serve > "${LOGS_DIR}/ollama.log" 2>&1 &

    echo $! >> "${PIDS_FILE}"
    echo -e "${GREEN}✓ Ollama iniciado (PID: $!)${NC}"

    # Esperar a que Ollama esté listo
    sleep 5

    # Descargar modelo llama3 en Ollama
    echo -e "${BLUE}Descargando modelo llama3 en Ollama...${NC}"
    singularity exec docker://ollama/ollama:latest ollama pull llama3

    # Iniciar Elasticsearch
    echo -e "${BLUE}Iniciando Elasticsearch...${NC}"
    nohup singularity run \
        --net \
        --network-args="--portmap=9200:9200/tcp" \
        --bind="${SCRIPT_DIR}/elasticsearch_data:/usr/share/elasticsearch/data" \
        --env="discovery.type=single-node" \
        --env="xpack.security.enabled=false" \
        --env="ES_JAVA_OPTS=-Xms512m -Xmx512m" \
        docker://docker.elastic.co/elasticsearch/elasticsearch:8.14.0 > "${LOGS_DIR}/elasticsearch.log" 2>&1 &

    echo $! >> "${PIDS_FILE}"
    echo -e "${GREEN}✓ Elasticsearch iniciado (PID: $!)${NC}"

    # Esperar a que Elasticsearch esté listo
    if ! wait_for_service "Elasticsearch" "http://localhost:9200"; then
        echo -e "${RED}No se pudo conectar a Elasticsearch. Verificando logs...${NC}"
        tail -20 "${LOGS_DIR}/elasticsearch.log"
        exit 1
    fi

    # Iniciar aplicación RAG
    echo -e "${BLUE}Iniciando aplicación RAG...${NC}"

    # Verificar si existe la imagen de Singularity
    if [ ! -f "rag-app.sif" ]; then
        echo -e "${YELLOW}Construyendo imagen de Singularity para RAG app...${NC}"
        singularity build rag-app.sif Singularity
    fi

    nohup singularity run \
        --net \
        --network-args="--portmap=8000:8000/tcp" \
        --bind="${SCRIPT_DIR}/app:/app" \
        rag-app.sif > "${LOGS_DIR}/rag-app.log" 2>&1 &

    echo $! >> "${PIDS_FILE}"
    echo -e "${GREEN}✓ Aplicación RAG iniciada (PID: $!)${NC}"

    echo -e "${GREEN}=== Sistema RAG desplegado exitosamente ===${NC}"
    echo -e "${GREEN}API disponible en: http://localhost:8000${NC}"
    echo -e "${YELLOW}PIDs guardados en: ${PIDS_FILE}${NC}"
    echo -e "${YELLOW}Logs en: ${LOGS_DIR}${NC}"
}

# Detener servicios
stop_services() {
    echo -e "${BLUE}=== Deteniendo Sistema RAG ===${NC}"

    if [ -f "${PIDS_FILE}" ]; then
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${YELLOW}Deteniendo proceso PID: ${pid}${NC}"
                kill "$pid"
                sleep 2
                if kill -0 "$pid" 2>/dev/null; then
                    echo -e "${YELLOW}Forzando terminación del proceso PID: ${pid}${NC}"
                    kill -9 "$pid"
                fi
                echo -e "${GREEN}✓ Proceso ${pid} detenido${NC}"
            else
                echo -e "${YELLOW}Proceso PID: ${pid} ya no está ejecutándose${NC}"
            fi
        done < "${PIDS_FILE}"
        rm -f "${PIDS_FILE}"
        echo -e "${GREEN}=== Todos los servicios detenidos ===${NC}"
    else
        echo -e "${YELLOW}No se encontró archivo de PIDs. Los servicios pueden no estar ejecutándose.${NC}"
    fi
}

# Mostrar estado de servicios
show_status() {
    echo -e "${BLUE}=== Estado del Sistema RAG ===${NC}"

    if [ -f "${PIDS_FILE}" ]; then
        echo -e "${GREEN}PIDs activos:${NC}"
        cat "${PIDS_FILE}"

        echo -e "\n${GREEN}Procesos ejecutándose:${NC}"
        while read -r pid; do
            if kill -0 "$pid" 2>/dev/null; then
                ps -p "$pid" -o pid,ppid,cmd,etime | tail -1
            fi
        done < "${PIDS_FILE}"
    else
        echo -e "${YELLOW}No hay servicios registrados${NC}"
    fi

    # Verificar servicios por puerto
    echo -e "\n${GREEN}Verificación de puertos:${NC}"
    if command -v netstat &> /dev/null; then
        netstat -tlnp 2>/dev/null | grep -E ':(11434|9200|8000)' || echo "netstat no disponible o puertos no encontrados"
    elif command -v ss &> /dev/null; then
        ss -tlnp | grep -E ':(11434|9200|8000)' || echo "ss no disponible o puertos no encontrados"
    else
        echo "Herramientas de red no disponibles"
    fi
}

# Mostrar logs de servicios
show_logs() {
    echo -e "${BLUE}=== Logs del Sistema RAG ===${NC}"

    if [ -d "${LOGS_DIR}" ]; then
        for log_file in "${LOGS_DIR}"/*.log; do
            if [ -f "$log_file" ]; then
                service_name=$(basename "$log_file" .log)
                echo -e "\n${YELLOW}=== Logs de ${service_name} ===${NC}"
                tail -20 "$log_file"
            fi
        done
    else
        echo -e "${YELLOW}No se encontró directorio de logs${NC}"
    fi
}

# Función principal
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
            echo "Uso: $0 [start|stop|restart|status|logs]"
            echo ""
            echo "Comandos:"
            echo "  start    - Iniciar todos los servicios"
            echo "  stop     - Detener todos los servicios"
            echo "  restart  - Reiniciar todos los servicios"
            echo "  status   - Mostrar estado de los servicios"
            echo "  logs     - Mostrar logs de los servicios"
            exit 1
            ;;
    esac
}

# Ejecutar función principal
main "$@"
