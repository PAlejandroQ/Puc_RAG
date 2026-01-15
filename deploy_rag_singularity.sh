#!/bin/bash
#SBATCH --job-name=RAG_Singularity_Deployment
#SBATCH --nodes=1
#SBATCH --nodelist=gpunode-1-3
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=rag_singularity_%j.out
#SBATCH --error=rag_singularity_%j.err

# =======================
# CONFIGURACIÓN DE RUTAS
# =======================
OLLAMA_SIF="/path/to/ollama.sif"       
ES_SIF="/path/to/elasticsearch.sif"    

OLLAMA_INSTANCE="ollama-rag-instance"
ES_INSTANCE="es-rag-instance"

OLLAMA_PORT=11434
ES_PORT=9200

# Parámetros AWS para el Túnel
AWS_IP="3.8.7.6"                        # <--- MODIFICA SI CAMBIA LA IP DE TU EC2
AWS_OLLAMA_PORT=11435
AWS_ES_PORT=9201
SSH_KEY_PATH="/path/to/my/key/aws_rag_key.pem"

# =======================
# FUNCIÓN DE LIMPIEZA
# =======================
cleanup() {
    echo "--- [TRAP] Iniciando limpieza de recursos ---"
    # Detener instancias Singularity
    singularity instance stop $OLLAMA_INSTANCE
    singularity instance stop $ES_INSTANCE

    # Detener túnel SSH
    SSH_PID=$(pgrep -f "ssh -N -i $SSH_KEY_PATH -R $AWS_OLLAMA_PORT:localhost:$OLLAMA_PORT -R $AWS_ES_PORT:localhost:$ES_PORT ec2-user@$AWS_IP")
    if [ ! -z "$SSH_PID" ]; then
        kill $SSH_PID
        echo "Túnel SSH detenido (PID $SSH_PID)"
    fi
    echo "--- Limpieza completa. ---"
}
trap cleanup EXIT TERM

# =======================
# CARGA DE MÓDULOS
# =======================
module load singularity
module load cuda/12.1 # Cambia la versión si tu clúster usa otra

echo "Módulos cargados. Nodo asignado: $HOSTNAME"

# =======================
# INICIO DE ELASTICSEARCH
# =======================
echo "Iniciando Elasticsearch en Singularity..."
singularity instance start \
    --bind /tmp:/tmp \
    --name $ES_INSTANCE \
    $ES_SIF \
    /usr/share/elasticsearch/bin/elasticsearch -Ediscovery.type=single-node -Expack.security.enabled=false &
sleep 20
echo "Elasticsearch iniciado."

# =======================
# INICIO DE OLLAMA (GPU)
# =======================
echo "Iniciando Ollama con soporte GPU..."
singularity instance start --nv \
    --bind /tmp:/tmp \
    --name $OLLAMA_INSTANCE \
    -p $OLLAMA_PORT:$OLLAMA_PORT \
    $OLLAMA_SIF \
    ollama serve &
sleep 10

echo "Descargando y preparando modelo de embedding (nomic-embed-text)..."
singularity exec instance://$OLLAMA_INSTANCE ollama run nomic-embed-text
echo "Ollama iniciado y modelo de embedding listo."

# =======================
# TÚNEL INVERSO SSH
# =======================
echo "Configurando túnel SSH reverso a AWS ($AWS_IP)..."
ssh -N -i $SSH_KEY_PATH \
    -R $AWS_OLLAMA_PORT:localhost:$OLLAMA_PORT \
    -R $AWS_ES_PORT:localhost:$ES_PORT \
    ec2-user@$AWS_IP &
echo "Túnel SSH activo."

echo "----------------------------------------------------------------------"
echo "TÚNEL ACTIVO. ACCESO DESDE AWS EC2:"
echo "  Ollama:        http://localhost:$AWS_OLLAMA_PORT"
echo "  Elasticsearch: http://localhost:$AWS_ES_PORT"
echo "----------------------------------------------------------------------"

# =======================
# MANTENER JOB ACTIVO
# =======================
echo "Manteniendo job SLURM activo. El tiempo restante es gestionado por SLURM."
wait