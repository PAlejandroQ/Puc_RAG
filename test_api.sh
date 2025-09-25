#!/bin/bash

# =============================================================================
# Script de Pruebas para API RAG con Singularity
# =============================================================================
# Este script proporciona ejemplos de uso de la API RAG una vez desplegada
# con Singularity.
#
# Uso: ./test_api.sh [health|query|interactive]
# =============================================================================

set -e

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

API_BASE_URL="http://localhost:8000"

# Verificar si la API está disponible
check_api() {
    echo -e "${BLUE}Verificando conectividad con la API RAG...${NC}"

    if curl -s "${API_BASE_URL}/" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API RAG está disponible en ${API_BASE_URL}${NC}"
        return 0
    else
        echo -e "${RED}✗ API RAG no está disponible en ${API_BASE_URL}${NC}"
        echo -e "${YELLOW}Asegúrese de que el sistema esté desplegado con: ./deploy.sh start${NC}"
        return 1
    fi
}

# Health check
health_check() {
    echo -e "${BLUE}=== Health Check ===${NC}"

    response=$(curl -s "${API_BASE_URL}/")

    echo -e "${GREEN}Respuesta del servidor:${NC}"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
}

# Consulta simple
query_document() {
    local question="$1"

    echo -e "${BLUE}=== Consulta: $question ===${NC}"

    response=$(curl -s -X POST "${API_BASE_URL}/query" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"$question\"}")

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Respuesta:${NC}"
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    else
        echo -e "${RED}Error al hacer la consulta${NC}"
        return 1
    fi
}

# Modo interactivo
interactive_mode() {
    echo -e "${BLUE}=== Modo Interactivo de Consultas ===${NC}"
    echo -e "${YELLOW}Escriba 'quit' o 'exit' para salir${NC}"
    echo -e "${YELLOW}Escriba 'clear' para limpiar la pantalla${NC}"
    echo ""

    while true; do
        echo -ne "${GREEN}Pregunta: ${NC}"
        read -r question

        if [[ "$question" == "quit" || "$question" == "exit" ]]; then
            echo -e "${YELLOW}¡Hasta luego!${NC}"
            break
        elif [[ "$question" == "clear" ]]; then
            clear
            continue
        elif [[ -z "$question" ]]; then
            continue
        fi

        query_document "$question"
        echo ""
    done
}

# Ejemplos predefinidos
run_examples() {
    echo -e "${BLUE}=== Ejecutando Ejemplos de Consulta ===${NC}"

    local examples=(
        "¿En qué caso es inadmisible la sucesión del cónyuge?"
        "¿Cuáles son los requisitos para la sucesión testamentaria?"
        "¿Qué es un contrato de compraventa?"
        "¿Cuáles son las obligaciones del vendedor en un contrato de compraventa?"
    )

    for question in "${examples[@]}"; do
        query_document "$question"
        echo -e "${YELLOW}Presione Enter para continuar...${NC}"
        read -r
    done
}

# Función principal
main() {
    if ! check_api; then
        exit 1
    fi

    case "${1:-interactive}" in
        health)
            health_check
            ;;
        query)
            if [ -z "$2" ]; then
                echo -e "${YELLOW}Uso: $0 query \"su pregunta aquí\"${NC}"
                exit 1
            fi
            query_document "$2"
            ;;
        examples)
            run_examples
            ;;
        interactive)
            interactive_mode
            ;;
        *)
            echo "Uso: $0 [health|query|examples|interactive]"
            echo ""
            echo "Comandos:"
            echo "  health      - Verificar estado de la API"
            echo "  query       - Hacer una consulta específica"
            echo "  examples    - Ejecutar ejemplos predefinidos"
            echo "  interactive - Modo interactivo de consultas"
            echo ""
            echo "Ejemplos:"
            echo "  $0 health"
            echo "  $0 query \"¿Qué es un contrato?\""
            echo "  $0 examples"
            echo "  $0 interactive"
            exit 1
            ;;
    esac
}

# Ejecutar función principal
main "$@"
