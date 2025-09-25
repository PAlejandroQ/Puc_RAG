# RAG System Deployment with Singularity

This document provides detailed instructions to deploy the RAG (Retrieval-Augmented Generation) system using Singularity in HPC environments, as a functional alternative to Docker Compose deployment.

## 📋 Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [System Architecture](#system-architecture)
- [Installation and Setup](#installation-and-setup)
- [Service Deployment](#service-deployment)
- [API Usage](#api-usage)
- [Service Management](#service-management)
- [Monitoring and Logs](#monitoring-and-logs)
- [Troubleshooting](#troubleshooting)
- [Docker vs Singularity Comparison](#docker-vs-singularity-comparison)

## 🎯 Overview

This RAG system allows you to query legal documents in natural language using:

- **Ollama**: Local language model (Llama 3) for answer generation
- **Elasticsearch**: Vector search engine for retrieving relevant documents
- **FastAPI**: REST API for system interaction
- **LangChain**: Framework to orchestrate the RAG pipeline

## 🖥️ System Requirements

### Minimum Recommended Hardware
- **CPU**: 4+ cores
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB available
- **Network**: Internet connectivity for initial model download

### Required Software
- **Singularity**: Version 3.7+
- **curl/wget**: For connectivity checks
- **bash**: Bash-compatible shell

### Requirements Check
```bash
# Check Singularity
singularity --version

# Check system resources
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | awk 'NR==2{printf "%.1fG", $2/1024}')"
echo "Disk: $(df -h . | awk 'NR==2{print $4}')"
```

## 🏗️ System Architecture

```
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│               │    │               │    │               │
│   User        │◄──►│   FastAPI RAG │◄──►│   Ollama LLM  │
│               │    │   Application │    │   (Llama 3)   │
└───────────────┘    └───────────────┘    └───────────────┘
                          │
                          ▼
                   ┌───────────────┐
                   │ Elasticsearch │
                   │  Vector Store │
                   └───────────────┘
```

### Port Mapping
- **11434**: Ollama API
- **9200**: Elasticsearch API
- **8000**: FastAPI RAG API

### Persistent Volumes
- `ollama_data`: Ollama models and configuration
- `elasticsearch_data`: Elasticsearch indices and data
- `app`: Application code and documents

## 🔧 Installation and Setup

### 1. Environment Preparation
```bash
# Clone the repository
git clone <repository-url>
cd Puc_RAG

# Make the deployment script executable
chmod +x deploy.sh

# Create directory structure
./deploy.sh start
```

### 2. Build the Singularity Image
```bash
# Build the RAG application image
singularity build rag-app.sif Singularity

# Verify the image was created
ls -lh rag-app.sif
```

### 3. Environment Variable Configuration
The system uses the following environment variables (set automatically):
- `ELASTICSEARCH_URL=http://localhost:9200`
- `OLLAMA_BASE_URL=http://localhost:11434`

## 🚀 Service Deployment

### Start the Full System
```bash
# Start all services
./deploy.sh start
```

This command:
1. ✅ Checks Singularity installation
2. ✅ Creates required directories
3. ✅ Starts Ollama and downloads the Llama 3 model
4. ✅ Starts Elasticsearch with optimized configuration
5. ✅ Waits for services to be ready
6. ✅ Starts the RAG application
7. ✅ Provides connection information

### Deployment Verification
```bash
# Check service status
./deploy.sh status

# View service logs
./deploy.sh logs
```

## 🔌 API Usage

Once all services are running, the API will be available at:
```
http://localhost:8000
```

### Available Endpoints

#### 1. Health Check
```bash
curl http://localhost:8000/
```

Expected response:
```json
{
  "message": "RAG Document Q&A API",
  "status": "running"
}
```

#### 2. Document Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "In what case is the spouse's succession inadmissible?"
  }'
```

Expected response:
```json
{
  "answer": "The spouse's succession is inadmissible when...",
  "source_documents": [
    {
      "content": "Relevant document text...",
      "metadata": {
        "source": "Codigo_Civil_split.pdf",
        "page": 1
      }
    }
  ]
}
```

### Example Queries
```bash
# Succession query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the requirements for testamentary succession?"}'

# Contract query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a sales contract?"}'
```

## ⚙️ Service Management

### Available States
```bash
# Start services
./deploy.sh start

# Stop services
./deploy.sh stop

# Restart services
./deploy.sh restart

# Check status
./deploy.sh status

# View logs
./deploy.sh logs
```

### Manual Service Management

#### Stop Individual Service
```bash
# View active PIDs
cat singularity_pids.txt

# Stop a specific service
kill <SERVICE_PID>

# Stop all services
kill $(cat singularity_pids.txt)
```

#### Connectivity Check
```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check Elasticsearch
curl http://localhost:9200/_cluster/health

# Check RAG API
curl http://localhost:8000/
```

## 📊 Monitoring and Logs

### Log Structure
Logs are organized in the `logs/` directory:
```
logs/
├── ollama.log          # Ollama logs
├── elasticsearch.log   # Elasticsearch logs
└── rag-app.log         # RAG application logs
```

### Real-Time Monitoring
```bash
# View all service logs
./deploy.sh logs

# View specific logs
tail -f logs/ollama.log
tail -f logs/elasticsearch.log
tail -f logs/rag-app.log

# Combine all logs
tail -f logs/*.log
```

### Systemd Logs (if applicable)
```bash
# View system logs
journalctl -u singularity-rag -f

# View logs for specific timestamps
journalctl --since="2024-01-01 00:00:00" --until="2024-01-02 00:00:00"
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Singularity is not installed
```bash
# On Ubuntu/Debian
sudo apt-get update
sudo apt-get install singularity-container

# On CentOS/RHEL
sudo yum install singularity
```

#### 2. Ports already in use
```bash
# Check which processes use the ports
netstat -tlnp | grep -E ':(11434|9200|8000)'

# Kill processes occupying the ports
sudo kill -9 <PID>
```

#### 3. Models not downloading
```bash
# Download model manually
singularity exec docker://ollama/ollama:latest ollama pull llama3

# List installed models
singularity exec docker://ollama/ollama:latest ollama list
```

#### 4. Elasticsearch does not start
```bash
# Check Elasticsearch logs
tail -50 logs/elasticsearch.log

# Check directory permissions
ls -la elasticsearch_data/

# Clean Elasticsearch data (if needed)
rm -rf elasticsearch_data/*
```

#### 5. RAG application not responding
```bash
# Check application logs
tail -50 logs/rag-app.log

# Check service connectivity
curl http://localhost:11434/api/tags
curl http://localhost:9200/_cluster/health

# Restart only the application
./deploy.sh restart
```

### Resource Check
```bash
# Monitor CPU and memory usage
htop

# View Singularity processes
ps aux | grep singularity

# Check disk usage
df -h
du -sh ollama_data/ elasticsearch_data/ app/
```

## ⚖️ Docker vs Singularity Comparison

| Aspect      | Docker           | Singularity      |
|------------|------------------|------------------|
| **Env**    | Servers, Cloud   | HPC, Clusters    |
| **User**   | root/sudo        | Normal user      |
| **Network**| Virtual          | Host-shared      |
| **Volumes**| Named volumes    | Bind mounts      |
| **Persistence** | Docker volumes | Host directories |
| **Security**| Namespaces       | Optional cgroups |
| **GPU**    | Native           | Specific config  |

### Singularity Advantages for this Project:
1. ✅ **No privileges**: Does not require sudo/root
2. ✅ **Shared network**: Direct communication between services
3. ✅ **HPC compatible**: Works on clusters and supercomputers
4. ✅ **Reproducible**: Consistent environments between runs
5. ✅ **Portable**: Easy migration between systems

## 📝 Important Notes

### Performance
- **Memory**: Elasticsearch requires at least 1GB RAM
- **CPU**: Ollama benefits from multiple cores
- **Network**: Low latency improves user experience

### Security
- Containers share the host network
- Data is stored in local directories
- No need to expose external ports

### Maintenance
- Ollama models are stored persistently
- Elasticsearch indices are kept between restarts
- Processed documents are not automatically re-indexed

### Backup and Recovery
```bash
# Backup data
tar -czf rag-backup-$(date +%Y%m%d).tar.gz ollama_data/ elasticsearch_data/

# Restore data
tar -xzf rag-backup-*.tar.gz
```

## 📞 Support

For specific issues:
1. Check logs: `./deploy.sh logs`
2. Check status: `./deploy.sh status`
3. Review the troubleshooting section
4. Contact the development team with full logs

---
*This RAG system is optimized for HPC environments and provides a robust alternative to traditional Docker deployment.*
