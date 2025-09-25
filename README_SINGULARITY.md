# Despliegue del Sistema RAG con Singularity

Este documento proporciona instrucciones detalladas para desplegar el sistema RAG (Retrieval-Augmented Generation) utilizando Singularity en entornos HPC, como alternativa funcional al despliegue con Docker Compose.

## 📋 Tabla de Contenidos

- [Visión General](#visión-general)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Instalación y Configuración](#instalación-y-configuración)
- [Despliegue de Servicios](#despliegue-de-servicios)
- [Uso de la API](#uso-de-la-api)
- [Gestión de Servicios](#gestión-de-servicios)
- [Monitoreo y Logs](#monitoreo-y-logs)
- [Solución de Problemas](#solución-de-problemas)
- [Comparación Docker vs Singularity](#comparación-docker-vs-singularity)

## 🎯 Visión General

Este sistema RAG permite hacer consultas en lenguaje natural sobre documentos legales utilizando:

- **Ollama**: Modelo de lenguaje local (Llama 3) para generación de respuestas
- **Elasticsearch**: Motor de búsqueda vectorial para recuperación de documentos relevantes
- **FastAPI**: API REST para interactuar con el sistema
- **LangChain**: Framework para orquestar el pipeline RAG

## 🖥️ Requisitos del Sistema

### Hardware Mínimo Recomendado
- **CPU**: 4+ núcleos
- **RAM**: 8GB (16GB recomendado)
- **Almacenamiento**: 10GB disponibles
- **Red**: Conectividad a internet para descarga inicial de modelos

### Software Requerido
- **Singularity**: Versión 3.7+
- **curl/wget**: Para verificaciones de conectividad
- **bash**: Shell compatible con bash

### Verificación de Requisitos
```bash
# Verificar Singularity
singularity --version

# Verificar recursos del sistema
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | awk 'NR==2{printf "%.1fG", $2/1024}')"
echo "Disk: $(df -h . | awk 'NR==2{print $4}')"
```

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   Usuario       │◄──►│   FastAPI RAG   │◄──►│   Ollama LLM    │
│                 │    │   Aplicación    │    │   (Llama 3)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │  Elasticsearch  │
                       │  Vector Store   │
                       └─────────────────┘
```

### Mapeo de Puertos
- **11434**: Ollama API
- **9200**: Elasticsearch API
- **8000**: FastAPI RAG API

### Volúmenes Persistentes
- `ollama_data`: Modelos y configuraciones de Ollama
- `elasticsearch_data`: Índices y datos de Elasticsearch
- `app`: Código de la aplicación y documentos

## 🔧 Instalación y Configuración

### 1. Preparación del Entorno
```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd Puc_RAG

# Dar permisos de ejecución al script
chmod +x deploy.sh

# Crear estructura de directorios
./deploy.sh start
```

### 2. Construcción de la Imagen de Singularity
```bash
# Construir la imagen de la aplicación RAG
singularity build rag-app.sif Singularity

# Verificar que la imagen se creó correctamente
ls -lh rag-app.sif
```

### 3. Configuración de Variables de Entorno
El sistema utiliza las siguientes variables de entorno (configuradas automáticamente):
- `ELASTICSEARCH_URL=http://localhost:9200`
- `OLLAMA_BASE_URL=http://localhost:11434`

## 🚀 Despliegue de Servicios

### Inicio Completo del Sistema
```bash
# Iniciar todos los servicios
./deploy.sh start
```

Este comando:
1. ✅ Verifica la instalación de Singularity
2. ✅ Crea directorios necesarios
3. ✅ Inicia Ollama y descarga el modelo Llama 3
4. ✅ Inicia Elasticsearch con configuración optimizada
5. ✅ Espera a que los servicios estén listos
6. ✅ Inicia la aplicación RAG
7. ✅ Proporciona información de conexión

### Verificación del Despliegue
```bash
# Verificar estado de los servicios
./deploy.sh status

# Ver logs de los servicios
./deploy.sh logs
```

## 🔌 Uso de la API

Una vez que todos los servicios estén ejecutándose, la API estará disponible en:
```
http://localhost:8000
```

### Endpoints Disponibles

#### 1. Health Check
```bash
curl http://localhost:8000/
```

Respuesta esperada:
```json
{
  "message": "RAG Document Q&A API",
  "status": "running"
}
```

#### 2. Consulta de Documentos
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿En qué caso es inadmisible la sucesión del cónyuge?"
  }'
```

Respuesta esperada:
```json
{
  "answer": "La sucesión del cónyuge es inadmisible cuando...",
  "source_documents": [
    {
      "content": "Texto relevante del documento...",
      "metadata": {
        "source": "Codigo_Civil_split.pdf",
        "page": 1
      }
    }
  ]
}
```

### Ejemplos de Consultas
```bash
# Consulta sobre sucesión
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Cuáles son los requisitos para la sucesión testamentaria?"}'

# Consulta sobre contratos
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Qué es un contrato de compraventa?"}'
```

## ⚙️ Gestión de Servicios

### Estados Disponibles
```bash
# Iniciar servicios
./deploy.sh start

# Detener servicios
./deploy.sh stop

# Reiniciar servicios
./deploy.sh restart

# Verificar estado
./deploy.sh status

# Ver logs
./deploy.sh logs
```

### Gestión Manual de Servicios

#### Detención Individual
```bash
# Ver PIDs activos
cat singularity_pids.txt

# Detener un servicio específico
kill <PID_DEL_SERVICIO>

# Detener todos los servicios
kill $(cat singularity_pids.txt)
```

#### Verificación de Conectividad
```bash
# Verificar Ollama
curl http://localhost:11434/api/tags

# Verificar Elasticsearch
curl http://localhost:9200/_cluster/health

# Verificar RAG API
curl http://localhost:8000/
```

## 📊 Monitoreo y Logs

### Estructura de Logs
Los logs se organizan en el directorio `logs/`:
```
logs/
├── ollama.log          # Logs de Ollama
├── elasticsearch.log   # Logs de Elasticsearch
└── rag-app.log         # Logs de la aplicación RAG
```

### Monitoreo en Tiempo Real
```bash
# Ver logs de todos los servicios
./deploy.sh logs

# Ver logs específicos
tail -f logs/ollama.log
tail -f logs/elasticsearch.log
tail -f logs/rag-app.log

# Combinar logs de todos los servicios
tail -f logs/*.log
```

### Logs de Systemd (si aplica)
```bash
# Ver logs del sistema
journalctl -u singularity-rag -f

# Ver logs con timestamps específicos
journalctl --since="2024-01-01 00:00:00" --until="2024-01-02 00:00:00"
```

## 🔍 Solución de Problemas

### Problemas Comunes

#### 1. Singularity no está instalado
```bash
# En Ubuntu/Debian
sudo apt-get update
sudo apt-get install singularity-container

# En CentOS/RHEL
sudo yum install singularity
```

#### 2. Puertos ya en uso
```bash
# Ver qué procesos usan los puertos
netstat -tlnp | grep -E ':(11434|9200|8000)'

# Matar procesos que ocupen los puertos
sudo kill -9 <PID>
```

#### 3. Modelos no se descargan
```bash
# Descargar modelo manualmente
singularity exec docker://ollama/ollama:latest ollama pull llama3

# Verificar modelos instalados
singularity exec docker://ollama/ollama:latest ollama list
```

#### 4. Elasticsearch no inicia
```bash
# Verificar logs de Elasticsearch
tail -50 logs/elasticsearch.log

# Verificar permisos de directorio
ls -la elasticsearch_data/

# Limpiar datos de Elasticsearch (si es necesario)
rm -rf elasticsearch_data/*
```

#### 5. Aplicación RAG no responde
```bash
# Verificar logs de la aplicación
tail -50 logs/rag-app.log

# Verificar conectividad a servicios
curl http://localhost:11434/api/tags
curl http://localhost:9200/_cluster/health

# Reiniciar solo la aplicación
./deploy.sh restart
```

### Verificación de Recursos
```bash
# Monitorear uso de CPU y memoria
htop

# Ver procesos de Singularity
ps aux | grep singularity

# Ver uso de disco
df -h
du -sh ollama_data/ elasticsearch_data/ app/
```

## ⚖️ Comparación Docker vs Singularity

| Aspecto | Docker | Singularity |
|---------|--------|-------------|
| **Entorno** | Servidores, Cloud | HPC, Clusters |
| **Usuario** | root/sudo | Usuario normal |
| **Red** | Virtual | Host compartida |
| **Volúmenes** | Named volumes | Bind mounts |
| **PERSISTENCIA** | Docker volumes | Directorios host |
| **Seguridad** | Namespaces | Cgroups opcional |
| **GPU Support** | Nativo | Configuración específica |

### Ventajas de Singularity para este proyecto:
1. ✅ **Sin privilegios**: No requiere sudo/root
2. ✅ **Red compartida**: Comunicación directa entre servicios
3. ✅ **HPC compatible**: Funciona en clusters y supercomputadoras
4. ✅ **Reproducible**: Entornos consistentes entre ejecuciones
5. ✅ **Portabilidad**: Fácil migración entre sistemas

## 📝 Notas Importantes

### Rendimiento
- **Memoria**: Elasticsearch requiere al menos 1GB RAM
- **CPU**: Ollama se beneficia de múltiples núcleos
- **Red**: Baja latencia mejora la experiencia del usuario

### Seguridad
- Los contenedores comparten la red del host
- Los datos se almacenan en directorios locales
- No se requiere exposición de puertos externos

### Mantenimiento
- Los modelos de Ollama se almacenan persistentemente
- Los índices de Elasticsearch se mantienen entre reinicios
- Los documentos procesados no se reindexan automáticamente

### Backup y Recuperación
```bash
# Backup de datos
tar -czf rag-backup-$(date +%Y%m%d).tar.gz ollama_data/ elasticsearch_data/

# Restaurar datos
tar -xzf rag-backup-*.tar.gz
```

## 📞 Soporte

Para problemas específicos:
1. Verificar logs: `./deploy.sh logs`
2. Verificar estado: `./deploy.sh status`
3. Revisar sección de solución de problemas
4. Contactar al equipo de desarrollo con logs completos

---
*Este sistema RAG está optimizado para entornos HPC y proporciona una alternativa robusta al despliegue tradicional con Docker.*
