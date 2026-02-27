# Inference Engineering

Proyecto educativo que implementa un sistema de inferencia con búsqueda semántica, RAG y observabilidad de producción. Construido con FastAPI, SentenceTransformers, ChromaDB, Redis, Ollama y un cliente en Go.

Diseñado para aprender **Inference Engineering** y **arquitectura de microservicios** de forma progresiva, fase por fase, con código de calidad profesional.

---

## Stack completo

```
┌─────────────────────────────────────────────────────────────────┐
│                         Cliente Go                               │
│                   (inference-client/main.go)                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP REST + Bearer Auth
     ┌─────────────────────┼──────────────────────────┐
     │                     │                          │
POST /embedding       POST /search              POST /rag
POST /ingest          GET  /health              GET  /metrics
                      GET  /metrics/prometheus
     │                     │                          │
     ▼                     ▼                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  FastAPI Inference Service :8000                  │
│                                                                  │
│  Auth (Bearer API key) → Rate Limit (60 req/min) → Handler       │
│                                                                  │
│  STARTUP (una sola vez, bloqueante):                             │
│    [1] load_model()     → SentenceTransformer all-MiniLM-L6-v2  │
│    [2] init_redis()     → cache L2 distribuido                   │
│    [3] init_store()     → ChromaDB HNSW index                    │
│    [4] precompute()     → indexa corpus seed (15 docs)           │
│    [5] pull_model()     → descarga modelo Ollama si falta        │
│                                                                  │
│  REQUEST TIME:                                                   │
│    /embedding → L1 cache → L2 Redis → model.encode()            │
│    /search    → embed(query) → ChromaDB HNSW search             │
│    /ingest    → embed(text) → ChromaDB upsert                   │
│    /rag       → search(top_k) → build_prompt() → ollama.chat()  │
└──────────┬──────────────────────┬───────────────────────────────┘
           │                      │
           ▼                      ▼
  ┌─────────────────┐    ┌─────────────────┐
  │   Redis :6379   │    │  Ollama :11434  │
  │  Cache L2       │    │  llama3.2:1b    │
  │  Rate Limiting  │    │  LLM local      │
  └─────────────────┘    └─────────────────┘
```

---

## Fases implementadas

| Fase | Qué agrega | Archivos clave |
|---|---|---|
| **Base** | FastAPI + embeddings + búsqueda semántica in-memory + cliente Go | `main.py`, `embedding_service.py`, `search_service.py` |
| **Fase 2** | ChromaDB (HNSW index persistente) + `POST /ingest` + `top_k` | `vector_store.py` |
| **Fase 3** | RAG completo con Ollama + `POST /rag` | `rag_service.py`, `ollama_client.py` |
| **Fase 4** | Cache L1 (dict) + L2 (Redis) en `embedding_service` | `redis_cache.py` |
| **Fase 5** | API key auth + rate limiting (sliding window) + Prometheus metrics | `auth.py`, `rate_limiter.py`, `prometheus_metrics.py` |

---

## Estructura de carpetas

```
.
├── README.md
├── docker-compose.yml
│
├── inference-service/          ← Servicio Python (FastAPI)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 ← Entry point, lifespan, routers, middleware
│   │
│   ├── services/
│   │   ├── embedding_service.py   ← Cache L1+L2 + único caller del modelo
│   │   ├── search_service.py      ← Corpus indexado + búsqueda semántica
│   │   ├── vector_store.py        ← ChromaDB client (HNSW, upsert, search)
│   │   ├── rag_service.py         ← Pipeline RAG: retrieve → augment → generate
│   │   ├── ollama_client.py       ← HTTP client para Ollama API
│   │   ├── redis_cache.py         ← Cache L2 distribuido (bytes, TTL, SHA256 keys)
│   │   ├── auth.py                ← Bearer token auth via API_KEYS env var
│   │   ├── rate_limiter.py        ← Sliding window rate limit via Redis ZSET
│   │   ├── health_service.py      ← Readiness flags por componente
│   │   └── metrics_store.py       ← Contadores in-memory (JSON /metrics)
│   │
│   ├── models/
│   │   ├── embedding_model.py     ← Singleton SentenceTransformer
│   │   └── schemas.py             ← Todos los contratos Pydantic
│   │
│   ├── utils/
│   │   ├── similarity.py          ← Cosine similarity pura (numpy)
│   │   ├── timer.py               ← Context manager de latencia
│   │   └── logger.py              ← JSON structured logging
│   │
│   └── data/
│       └── documents.py           ← 15 documentos ML/AI (corpus seed)
│
└── inference-client/           ← Cliente Go (sin dependencias externas)
    ├── go.mod
    └── main.go
```

---

## Cómo correrlo

### Prerrequisitos

- Docker >= 24.0
- Docker Compose >= 2.0
- Go >= 1.22 (solo para el cliente Go)

### Stack completo (recomendado)

```bash
# Primera vez: build + descarga modelo HF (~90MB) + descarga modelo Ollama (~1.3GB)
docker compose up --build

# Siguientes veces: modelos cacheados en volúmenes (~15-30s startup)
docker compose up
```

### Solo core (sin RAG, más rápido)

```bash
# Levanta solo inference-service + Redis, sin Ollama
docker compose up inference-service redis --build
```

### Verificar que está listo

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "corpus_initialized": true,
  "cache_ready": true,
  "redis_connected": true,
  "ollama_ready": true
}
```

Verás en los logs de startup:
```json
{"event": "redis_connected", "url": "redis://redis:6379"}
{"event": "vector_store_init_completed", "existing_docs": 0, "duration_ms": 45.2}
{"event": "model_loading_completed", "duration_ms": 2341.5}
{"event": "corpus_index_completed", "indexed": 15, "duration_ms": 823.2}
{"event": "ollama_ready", "model": "llama3.2:1b"}
{"event": "application_startup_completed", "ready": true, "corpus_size": 15, "cache_size": 15}
```

### Detener

```bash
docker compose down          # detiene contenedores, preserva volúmenes
docker compose down -v       # detiene + borra volúmenes (re-descarga todo)
```

---

## Endpoints API

### Inference (requieren auth si `API_KEYS` está configurado)

| Método | Path | Descripción |
|---|---|---|
| `POST` | `/embedding` | Genera embedding de un texto (384 dims, cache-first) |
| `POST` | `/search` | Búsqueda semántica en ChromaDB, soporta `top_k` |
| `POST` | `/ingest` | Indexa un documento en ChromaDB (upsert) |
| `POST` | `/rag` | RAG completo: retrieval + generación con Ollama |

### Observabilidad (sin auth)

| Método | Path | Descripción |
|---|---|---|
| `GET` | `/health` | Readiness por componente (200/503) |
| `GET` | `/metrics` | Snapshot JSON: latencias, cache ratio |
| `GET` | `/metrics/prometheus` | Métricas Prometheus (Counters, Histograms, Gauges) |
| `GET` | `/docs` | Swagger UI interactiva |

---

## Ejemplos de uso

### Búsqueda semántica

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is backpropagation?", "top_k": 3}'
```

```json
{
  "result": "Backpropagation is an algorithm used to train neural networks...",
  "score": 0.8923,
  "results": [
    {"id": "doc_002", "text": "Backpropagation is...", "score": 0.8923},
    {"id": "doc_001", "text": "A neural network is...", "score": 0.7341},
    {"id": "doc_006", "text": "A transformer is...", "score": 0.6812}
  ],
  "elapsed_ms": 51.3
}
```

### Ingestar un documento nuevo

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"id": "doc_custom_001", "text": "Gradient descent minimizes the loss function..."}'
```

```json
{"id": "doc_custom_001", "total_documents": 16, "elapsed_ms": 48.2}
```

### RAG completo

```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "How does backpropagation work?", "top_k": 3}'
```

```json
{
  "answer": "Backpropagation works by computing gradients of the loss function...",
  "sources": [
    {"id": "doc_002", "text": "Backpropagation is an algorithm...", "score": 0.8923}
  ],
  "model": "llama3.2:1b",
  "retrieve_ms": 52.1,
  "generate_ms": 3420.5,
  "total_ms": 3474.2
}
```

### Generar embedding

```bash
curl -X POST http://localhost:8000/embedding \
  -H "Content-Type: application/json" \
  -d '{"text": "machine learning is powerful"}'
```

```json
{
  "embedding": [0.123, -0.456, ...],
  "dimension": 384,
  "cache_hit": false,
  "elapsed_ms": 47.2
}
```

### Métricas Prometheus

```bash
curl http://localhost:8000/metrics/prometheus
```

```
# HELP inference_requests_total Total number of HTTP requests
# TYPE inference_requests_total counter
inference_requests_total{endpoint="/search",method="POST",status_code="200"} 42.0

# HELP inference_request_duration_seconds HTTP request latency in seconds
# TYPE inference_request_duration_seconds histogram
inference_request_duration_seconds_bucket{endpoint="/search",le="0.05"} 35.0
inference_request_duration_seconds_bucket{endpoint="/search",le="0.1"} 41.0

# HELP inference_cache_hits_total Total embedding cache hits
# TYPE inference_cache_hits_total counter
inference_cache_hits_total{level="l1"} 38.0
inference_cache_hits_total{level="l2"} 3.0
```

### Con autenticación habilitada

```bash
# En docker-compose, setear: API_KEYS=mi-secret-key-32chars

curl -X POST http://localhost:8000/search \
  -H "Authorization: Bearer mi-secret-key-32chars" \
  -H "Content-Type: application/json" \
  -d '{"query": "what is a transformer?"}'
```

### Cliente Go

```bash
cd inference-client

go run main.go
go run main.go -query "explain the attention mechanism in transformers"
go run main.go -url http://localhost:8000 -query "what is a vector database?"
```

---

## Conceptos de Inference Engineering que aprenderás

### 1. Startup vs Request Time
El modelo se carga **una vez** en startup. Los embeddings del corpus se precomputan **una vez** y se indexan en ChromaDB. En runtime, cada `/search` genera **exactamente 1** embedding (solo la query).

### 2. Cache de dos niveles (L1 + L2)
Los embeddings son deterministas. L1 (dict in-memory, sub-μs) + L2 (Redis, ~0.5ms) evitan recomputar. Si Redis cae, L1 sigue funcionando. En la segunda request con la misma query: 0ms de inferencia.

### 3. Vector Store con HNSW
ChromaDB usa HNSW (Hierarchical Navigable Small World) para búsqueda vectorial en O(log n). En Fase 1 era O(n) brute-force. Con 1M documentos, la diferencia es de horas vs milisegundos.

### 4. RAG Pipeline
Retrieve → Augment → Generate. El retriever (ChromaDB) encuentra contexto relevante. El prompt se construye con ese contexto. El LLM genera una respuesta fundamentada. La calidad de la respuesta depende directamente de la calidad del retrieval.

### 5. Observabilidad
`/health` → readiness probe por componente. `/metrics` → JSON para desarrollo. `/metrics/prometheus` → Histograms con P50/P95/P99 para producción. Sin esto, un sistema ML en producción es una caja negra.

### 6. Auth y Rate Limiting
API keys como Bearer tokens. Rate limiting con sliding window en Redis ZSET: evita el burst problem de fixed windows. Fail-open si Redis no está disponible.

### 7. Contratos entre microservicios
El cliente Go modela los mismos schemas que el servidor Python, pero en Go. Cambiar la API sin actualizar el cliente resulta en error de compilación. Esto es lo que hace útil el tipado estricto en microservicios.

### 8. Graceful Degradation
Redis, Ollama y FAISS son opcionales en el startup. Si fallan, `/search` y `/embedding` siguen funcionando. Solo `/rag` requiere Ollama. Solo el rate limiting requiere Redis. El sistema degrada sin caerse.

---

## Separación startup vs request time

| Paso | Fase | Costo | Dónde |
|---|---|---|---|
| Cargar SentenceTransformer | Startup | ~2-3s | `embedding_model.load_model()` |
| Inicializar Redis | Startup | ~50ms | `redis_cache.init_redis()` |
| Inicializar ChromaDB | Startup | ~100ms | `vector_store.init_store()` |
| Indexar corpus (15 docs) | Startup | ~800ms | `search_service.precompute()` |
| Descargar modelo Ollama | Startup (1ra vez) | ~2-5min | `ollama_client.pull_model()` |
| **embed(query)** | **Request** | **~50ms (miss) / 0ms (hit)** | `embedding_service.embed()` |
| **HNSW search** | **Request** | **~3ms** | `vector_store.search()` |
| **LLM generate** | **Request** | **~3-30s (CPU)** | `ollama_client.generate()` |

---

## Stack tecnológico

| Componente | Tecnología | Versión |
|---|---|---|
| API Framework | FastAPI | 0.115.5 |
| Embeddings | sentence-transformers | 3.3.1 |
| Modelo de embeddings | all-MiniLM-L6-v2 | 384 dims |
| Vector Store | ChromaDB | 0.5.23 |
| LLM local | Ollama + llama3.2:1b | latest |
| Cache distribuido | Redis | 7-alpine |
| Métricas | prometheus-client | 0.21.1 |
| Validación | Pydantic | 2.10.3 |
| ASGI Server | Uvicorn | 0.32.1 |
| Numerical | NumPy | 1.26.4 |
| Cliente | Go | 1.22 |
| Contenedor | Docker | >= 24.0 |

---

## Variables de entorno

| Variable | Default | Descripción |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | URL del servicio Ollama |
| `OLLAMA_MODEL` | `llama3.2:1b` | Modelo LLM a usar |
| `RAG_TOP_K` | `3` | Documentos de contexto para RAG |
| `REDIS_URL` | `redis://redis:6379` | URL de Redis |
| `REDIS_EMBEDDING_TTL` | `86400` | TTL de embeddings en Redis (segundos) |
| `API_KEYS` | `` (vacío) | Keys separadas por coma. Vacío = auth desactivada |
| `RATE_LIMIT_REQUESTS` | `60` | Requests máximas por ventana |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Tamaño de la ventana de rate limiting |
| `HF_HOME` | `/app/.cache/huggingface` | Cache de modelos HuggingFace |

---

## Evolución futura

Este proyecto está diseñado como base para escalar hacia un sistema de producción real:

**Siguiente paso natural:**
- Reemplazar `redis_cache.py` con un cluster Redis para alta disponibilidad
- Reemplazar Ollama con vLLM para throughput de producción (batching, tensor parallelism)
- Agregar Grafana + Prometheus server para dashboards de latencia P95/P99
- Múltiples workers Uvicorn con estado compartido en Redis

**Camino a producción:**
- Deploy en Railway, Fly.io o AWS ECS con las variables de entorno configuradas
- CI/CD con GitHub Actions que corra linting + tests antes de cada push
- Secrets management via AWS Secrets Manager o Vault (en lugar de env vars planas)
