#!/bin/bash

# =============================================================================
# Test Script for RAG API with Singularity
# =============================================================================
# This script provides usage examples for the RAG API once deployed
# with Singularity.
#
# Usage: ./test_api.sh [health|query|interactive]
# =============================================================================

set -e

# Output colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

API_BASE_URL="http://localhost:8000"

# Check if the API is available
check_api() {
    echo -e "${BLUE}Checking connectivity with the RAG API...${NC}"

    if curl -s "${API_BASE_URL}/" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ RAG API is available at ${API_BASE_URL}${NC}"
        return 0
    else
        echo -e "${RED}✗ RAG API is not available at ${API_BASE_URL}${NC}"
        echo -e "${YELLOW}Make sure the system is deployed with: ./deploy.sh start${NC}"
        return 1
    fi
}

# Health check
health_check() {
    echo -e "${BLUE}=== Health Check ===${NC}"

    response=$(curl -s "${API_BASE_URL}/")

    echo -e "${GREEN}Server response:${NC}"
    echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
}

# Simple query
query_document() {
    local question="$1"

    echo -e "${BLUE}=== Query: $question ===${NC}"

    response=$(curl -s -X POST "${API_BASE_URL}/query" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"$question\"}")

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Response:${NC}"
        echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"
    else
        echo -e "${RED}Error making the query${NC}"
        return 1
    fi
}

# Interactive mode
interactive_mode() {
    echo -e "${BLUE}=== Interactive Query Mode ===${NC}"
    echo -e "${YELLOW}Type 'quit' or 'exit' to leave${NC}"
    echo -e "${YELLOW}Type 'clear' to clear the screen${NC}"
    echo ""

    while true; do
        echo -ne "${GREEN}Question: ${NC}"
        read -r question

        if [[ "$question" == "quit" || "$question" == "exit" ]]; then
            echo -e "${YELLOW}Goodbye!${NC}"
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

# Predefined examples
run_examples() {
    echo -e "${BLUE}=== Running Example Queries ===${NC}"

    local examples=(
        "In what case is the spouse's succession inadmissible?"
        "What are the requirements for testamentary succession?"
        "What is a sales contract?"
        "What are the obligations of the seller in a sales contract?"
    )

    for question in "${examples[@]}"; do
        query_document "$question"
        echo -e "${YELLOW}Press Enter to continue...${NC}"
        read -r
    done
}

# Main function
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
                echo -e "${YELLOW}Usage: $0 query \"your question here\"${NC}"
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
            echo "Usage: $0 [health|query|examples|interactive]"
            echo ""
            echo "Commands:"
            echo "  health      - Check API status"
            echo "  query       - Make a specific query"
            echo "  examples    - Run predefined examples"
            echo "  interactive - Interactive query mode"
            echo ""
            echo "Examples:"
            echo "  $0 health"
            echo "  $0 query \"What is a contract?\""
            echo "  $0 examples"
            echo "  $0 interactive"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
