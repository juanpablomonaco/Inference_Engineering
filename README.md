# Inference Engineering

Sistema de inferencia con búsqueda semántica, RAG, observabilidad y seguridad de producción. Construido con FastAPI, SentenceTransformers, ChromaDB, Redis, Ollama/vLLM, Prometheus y Grafana.

Diseñado para aprender **Inference Engineering** de forma progresiva, fase por fase, con código de calidad profesional.

---

## Stack completo

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CI/CD (GitHub Actions)                             │
│              lint (ruff) → test (pytest) → docker build              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│                         Cliente Go                                    │
│                   (inference-client/main.go)                          │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ HTTP REST + Bearer Auth
     ┌─────────────────────┼──────────────────────┐
     │                     │                      │
POST /embedding       POST /search          POST /rag
POST /ingest          GET  /health          GET  /metrics
                      GET  /metrics/prometheus
     │                     │                      │
     ▼                     ▼                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│             FastAPI Inference Service :8000                           │
│                                                                       │
│  Auth (Bearer API key) → Rate Limit (sliding window) → Handler        │
│                                                                       │
│  STARTUP [1] load_model()   → SentenceTransformer all-MiniLM-L6-v2   │
│          [2] init_redis()   → cache L2 distribuido                    │
│          [3] init_store()   → ChromaDB HNSW index                     │
│          [4] precompute()   → indexa corpus seed (15 docs)            │
│          [5] pull_model()   → descarga modelo LLM si falta            │
│                                                                       │
│  REQUEST  /embedding → L1 dict → L2 Redis → model.encode()           │
│           /search    → embed(query) → ChromaDB HNSW                  │
│           /ingest    → embed(text) → ChromaDB upsert                 │
│           /rag       → search(k) → build_prompt() → LLM.generate()   │
└──────────┬────────────────────────┬────────────────────────┬─────────┘
           │                        │                        │
           ▼                        ▼                        ▼
  ┌─────────────────┐   ┌──────────────────────┐   ┌─────────────────┐
  │   Redis :6379   │   │   Ollama :11434       │   │ Prometheus :9090│
  │  Cache L2       │   │   llama3.2:1b (dev)   │   │ + Grafana :3000 │
  │  Rate Limiting  │   │   vLLM (producción)   │   │ Dashboards P95  │
  └─────────────────┘   └──────────────────────┘   └─────────────────┘
```

---

## Fases implementadas

| Fase | Qué agrega | Archivos clave |
|---|---|---|
| **Base** | FastAPI + embeddings + búsqueda semántica + cliente Go | `embedding_service.py`, `search_service.py` |
| **2** | ChromaDB HNSW (O(log n)) + `POST /ingest` + `top_k` | `vector_store.py` |
| **3** | RAG pipeline con Ollama + `POST /rag` | `rag_service.py`, `ollama_client.py` |
| **4** | Cache L1 (dict) + L2 (Redis) en embedding | `redis_cache.py` |
| **5** | Bearer auth + rate limiting (sliding window) + Prometheus | `auth.py`, `rate_limiter.py`, `prometheus_metrics.py` |
| **6** | Grafana + Prometheus server + multi-worker + vLLM adapter + CI/CD | `vllm_client.py`, `.github/workflows/ci.yml`, `observability/` |

---

## Estructura de carpetas

```
.
├── README.md
├── docker-compose.yml
├── .env.example              ← plantilla de variables de entorno
├── .gitignore
│
├── .github/
│   └── workflows/
│       └── ci.yml            ← lint + test + docker build en cada push
│
├── observability/
│   ├── prometheus/
│   │   └── prometheus.yml    ← scrape config (inference-service + redis)
│   └── grafana/
│       ├── dashboards/
│       │   └── inference_service.json  ← dashboard precargado P50/P95/P99
│       └── provisioning/
│           ├── datasources/  ← Prometheus como datasource automático
│           └── dashboards/   ← carga automática del dashboard
│
├── inference-service/        ← Servicio Python (FastAPI)
│   ├── Dockerfile            ← multi-worker via WORKERS env var
│   ├── requirements.txt
│   ├── pytest.ini
│   ├── main.py               ← entry point, lifespan, routers, middleware
│   │
│   ├── services/
│   │   ├── embedding_service.py   ← cache L1+L2, único caller del modelo
│   │   ├── search_service.py      ← corpus indexado + búsqueda HNSW
│   │   ├── vector_store.py        ← ChromaDB client
│   │   ├── rag_service.py         ← pipeline RAG + backend swappeable
│   │   ├── ollama_client.py       ← cliente Ollama (dev)
│   │   ├── vllm_client.py         ← cliente vLLM (producción, OpenAI API)
│   │   ├── redis_cache.py         ← cache L2 (bytes, TTL, SHA256 keys)
│   │   ├── auth.py                ← Bearer token via API_KEYS
│   │   ├── rate_limiter.py        ← sliding window via Redis ZSET
│   │   ├── prometheus_metrics.py  ← Counters, Histograms, Gauges
│   │   ├── health_service.py      ← readiness flags por componente
│   │   └── metrics_store.py       ← contadores JSON /metrics
│   │
│   ├── models/
│   │   ├── embedding_model.py     ← singleton SentenceTransformer
│   │   └── schemas.py             ← contratos Pydantic de todos los endpoints
│   │
│   ├── utils/
│   │   ├── similarity.py          ← cosine similarity pura (numpy)
│   │   ├── timer.py               ← context manager de latencia
│   │   └── logger.py              ← JSON structured logging
│   │
│   ├── data/
│   │   └── documents.py           ← corpus seed (15 docs ML/AI)
│   │
│   └── tests/
│       ├── test_similarity.py     ← 11 tests cosine similarity + top_k
│       ├── test_schemas.py        ← 12 tests validación Pydantic
│       ├── test_timer.py          ← 3 tests context manager
│       └── test_metrics_store.py  ← 9 tests contadores in-memory
│
└── inference-client/         ← Cliente Go (sin dependencias externas)
    ├── go.mod
    └── main.go
```

---

## Cómo correrlo

### Prerrequisitos

- Docker >= 24.0 y Docker Compose >= 2.0
- Go >= 1.22 (solo para el cliente Go)

### Setup inicial

```bash
# Copiar plantilla de variables de entorno
cp .env.example .env
# Editar .env si querés personalizar (opcional para desarrollo)
```

### Stack completo (con observabilidad y RAG)

```bash
# Primera vez: build + descarga modelo HF (~90MB) + modelo Ollama (~1.3GB)
docker compose up --build

# Siguientes veces: todo cacheado en volúmenes
docker compose up
```

### Solo core (inferencia + Redis, sin RAG ni Grafana)

```bash
docker compose up inference-service redis --build
```

### Con observabilidad pero sin RAG

```bash
docker compose up inference-service redis prometheus grafana redis-exporter
```

### Multi-worker (para carga alta)

```bash
# Setear en .env o como env var:
WORKERS=4 docker compose up --build
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

### Detener

```bash
docker compose down          # preserva volúmenes (modelos cacheados)
docker compose down -v       # borra todo (re-descarga en el próximo up)
```

---

## Accesos

| Servicio | URL | Credenciales |
|---|---|---|
| API REST | http://localhost:8000 | — |
| Swagger UI | http://localhost:8000/docs | — |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| Ollama | http://localhost:11434 | — |
| Redis | localhost:6379 | — |

---

## Endpoints API

### Inference

| Método | Path | Auth | Descripción |
|---|---|---|---|
| `POST` | `/embedding` | ✓ | Genera embedding de un texto (384 dims, cache L1+L2) |
| `POST` | `/search` | ✓ | Búsqueda semántica HNSW en ChromaDB, soporta `top_k` |
| `POST` | `/ingest` | ✓ | Indexa un documento en ChromaDB (upsert) |
| `POST` | `/rag` | ✓ | RAG completo: retrieval + generación con Ollama o vLLM |

### Observabilidad

| Método | Path | Auth | Descripción |
|---|---|---|---|
| `GET` | `/health` | — | Readiness por componente (200/503) |
| `GET` | `/metrics` | — | Snapshot JSON: latencias, cache ratio |
| `GET` | `/metrics/prometheus` | — | Prometheus text exposition (Counters, Histograms, Gauges) |
| `GET` | `/docs` | — | Swagger UI |

✓ = requiere `Authorization: Bearer <api_key>` si `API_KEYS` está configurado

---

## Ejemplos rápidos

```bash
# Búsqueda semántica (top-3)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is backpropagation?", "top_k": 3}'

# Ingestar documento nuevo
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"id": "doc_custom", "text": "Gradient descent minimizes the loss..."}'

# RAG completo
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{"query": "How does backpropagation work?", "top_k": 3}'

# Con auth habilitada (API_KEYS=my-secret-key en .env)
curl -X POST http://localhost:8000/search \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "what is a transformer?"}'
```

### Cliente Go

```bash
cd inference-client
go run main.go
go run main.go -query "explain the attention mechanism"
go run main.go -url http://localhost:8000 -query "what is a vector database?"
```

---

## Dashboard Grafana

Al abrir http://localhost:3000 (admin/admin), encontrás el dashboard **"Inference Engineering"** precargado con:

| Panel | Métrica |
|---|---|
| Request Rate | requests/segundo por endpoint |
| Request Latency P50/P95/P99 | `/search` latencia en percentiles |
| Embedding Latency P50/P95/P99 | solo cache misses (inferencia real) |
| RAG Pipeline Latency P50/P95/P99 | retrieve + generate total |
| Search Latency P50/P95/P99 | solo ChromaDB HNSW search |
| Cache Hit Ratio | gauge 0-100% (rojo/amarillo/verde) |
| Cache Hits by Level | L1 vs L2 vs misses en tiempo real |
| Corpus Size | documentos en ChromaDB |
| System Status | model/redis/ollama UP o DOWN |
| HTTP Error Rate | 4xx y 5xx por endpoint |

---

## CI/CD — GitHub Actions

El workflow `.github/workflows/ci.yml` corre automáticamente en cada `push` a `main`/`develop` y en cada PR:

```
push/PR
  ↓
[1] lint      ruff check + ruff format --check
  ↓ (si pasa)
[2] test       pytest tests/ con 35+ assertions
  ↓ (si pasa)
[3] docker     docker buildx build (sin push, solo verifica que buildea)
```

Los tests corren sin servicios externos (sin modelo ML, sin Redis, sin Ollama).

---

## Cambiar backend LLM (Ollama → vLLM)

```bash
# En .env:
INFERENCE_BACKEND=vllm
VLLM_BASE_URL=http://vllm:8001
OLLAMA_MODEL=meta-llama/Llama-3.1-8B-Instruct

# Correr vLLM (requiere GPU con 8GB+ VRAM):
docker run --gpus all -p 8001:8000 vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max-model-len 4096
```

Sin cambiar ningún otro archivo. El swapeo es transparente para `rag_service.py`.

---

## Variables de entorno

Ver `.env.example` para la lista completa con descripciones y valores por defecto.

| Variable | Default | Descripción |
|---|---|---|
| `WORKERS` | `1` | Workers Uvicorn (producción: 2×CPUs+1) |
| `API_KEYS` | `` | Keys separadas por coma. Vacío = sin auth |
| `RATE_LIMIT_REQUESTS` | `60` | Max requests por ventana |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Tamaño de ventana (segundos) |
| `REDIS_URL` | `redis://redis:6379` | URL de Redis |
| `INFERENCE_BACKEND` | `ollama` | `ollama` o `vllm` |
| `OLLAMA_MODEL` | `llama3.2:1b` | Modelo LLM |
| `RAG_TOP_K` | `3` | Documentos de contexto para RAG |
| `GRAFANA_PASSWORD` | `admin` | Cambiar en producción |

---

## Conceptos aprendidos

| Concepto | Dónde está en el código |
|---|---|
| Startup vs Request Time | `main.py` lifespan, `embedding_model.load_model()` |
| Cache L1+L2 (dict + Redis) | `embedding_service.py`, `redis_cache.py` |
| Vector store HNSW (O(log n)) | `vector_store.py` → ChromaDB |
| RAG pipeline | `rag_service.py` |
| LLM backend swappeable | `vllm_client.py`, `ollama_client.py`, `INFERENCE_BACKEND` |
| Prometheus Histograms (P95) | `prometheus_metrics.py`, dashboard Grafana |
| Bearer auth + FastAPI Depends | `auth.py` |
| Sliding window rate limiting | `rate_limiter.py` → Redis ZSET |
| Graceful degradation | Redis/Ollama opcionales en startup |
| Multi-worker stateless | `WORKERS` env var, estado compartido en Redis |
| CI/CD | `.github/workflows/ci.yml` |
| Secrets management | `.env.example`, `.gitignore` |

---

## Stack tecnológico

| Componente | Tecnología | Versión |
|---|---|---|
| API | FastAPI + Uvicorn | 0.115.5 / 0.32.1 |
| Embeddings | sentence-transformers | 3.3.1 |
| Modelo embeddings | all-MiniLM-L6-v2 | 384 dims |
| Vector store | ChromaDB (HNSW) | 0.5.23 |
| LLM dev | Ollama + llama3.2:1b | latest |
| LLM prod | vLLM (OpenAI API) | latest |
| Cache | Redis | 7-alpine |
| Métricas | prometheus-client | 0.21.1 |
| Dashboards | Grafana | 11.3.1 |
| Scraping | Prometheus | 2.55.1 |
| Validación | Pydantic | 2.10.3 |
| Numérico | NumPy + faiss-cpu | 1.26.4 / 1.9.0 |
| Cliente | Go | 1.22 |
| Linter | ruff | 0.8.4 |
| Tests | pytest | latest |
| Contenedor | Docker | >= 24.0 |
