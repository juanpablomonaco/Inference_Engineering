# Inference Engineering — Semantic Search Microservice

Proyecto educativo que implementa un servicio de inferencia con búsqueda semántica usando FastAPI, SentenceTransformers y un cliente en Go. Diseñado para aprender **Inference Engineering** y **arquitectura de microservicios** con código de calidad profesional.

---

## Descripción

Este proyecto implementa dos componentes que se comunican vía HTTP:

- **inference-service** (Python): API de inferencia que genera embeddings y realiza búsqueda semántica sobre un corpus de documentos
- **inference-client** (Go): Cliente que consume el servicio, demuestra cómo los microservicios definen contratos explícitos entre sí

El foco está en **entender el flujo de inferencia real**: cómo un modelo ML se carga, cómo los vectores se comparan, y cómo se mide la performance de un sistema en producción.

---

## Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                    Cliente Go                        │
│             (inference-client/main.go)               │
└────────────────────┬────────────────────────────────┘
                     │ HTTP REST (JSON)
          ┌──────────┼──────────┐
          │          │          │
    POST /embedding  POST /search  GET /health + /metrics
          │          │          │
          ▼          ▼          ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Inference Service               │
│                                                      │
│  STARTUP (una sola vez):                             │
│    load_model() → precompute_corpus() → init cache   │
│                                                      │
│  REQUEST TIME:                                       │
│    /embedding → cache lookup → model.encode()        │
│    /search    → embed(query) → cosine similarity     │
│    /health    → flags booleanos                      │
│    /metrics   → contadores en memoria                │
└─────────────────────────────────────────────────────┘
```

### Separación startup vs request time

La decisión más importante del diseño:

| Fase | Qué ocurre | Costo |
|---|---|---|
| **Startup** | Cargar modelo, precomputar corpus, llenar cache | Una sola vez (~10s) |
| **Request** | Solo embed de la query + similarity | ~50ms |

En v1 (diseño naive), cada `/search` generaría N+1 embeddings.  
En este proyecto, cada `/search` genera **exactamente 1** embedding.

---

## Estructura de carpetas

```
inference-engineering/
│
├── README.md
├── docker-compose.yml
│
├── inference-service/          ← Servicio Python (FastAPI)
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                 ← Entry point + lifespan + routers
│   │
│   ├── services/
│   │   ├── embedding_service.py   ← Cache + embedding (único acceso al modelo)
│   │   ├── search_service.py      ← Corpus precomputado + búsqueda
│   │   ├── health_service.py      ← Flags de readiness
│   │   └── metrics_store.py       ← Contadores in-memory
│   │
│   ├── models/
│   │   ├── embedding_model.py     ← Singleton del modelo ML
│   │   └── schemas.py             ← Contratos de datos (Pydantic)
│   │
│   ├── utils/
│   │   ├── similarity.py          ← Cosine similarity (función pura)
│   │   ├── timer.py               ← Context manager de latencia
│   │   └── logger.py              ← Logging estructurado JSON
│   │
│   └── data/
│       └── documents.py           ← Corpus seed (15 documentos ML/AI)
│
└── inference-client/           ← Cliente Go
    ├── go.mod
    └── main.go
```

---

## Cómo correrlo

### Prerrequisitos

- Docker >= 24.0
- Docker Compose >= 2.0
- Go >= 1.22 (solo para el cliente)

### 1. Levantar el servicio

```bash
# Desde la raíz del proyecto
cd inference-engineering

# Primera vez (build de imagen + descarga del modelo)
docker-compose up --build

# Siguientes veces (usa imagen y modelo cacheados)
docker-compose up
```

Verás en los logs:
```json
{"event": "model_loading_started", "model": "sentence-transformers/all-MiniLM-L6-v2"}
{"event": "model_loading_completed", "duration_ms": 2341.5}
{"event": "corpus_precompute_completed", "total_documents": 15, "duration_ms": 823.2}
{"event": "application_startup_completed", "ready": true, "cache_size": 15}
```

### 2. Verificar que está listo

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "corpus_initialized": true,
  "cache_ready": true
}
```

### 3. Probar los endpoints

**Búsqueda semántica:**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what is backpropagation?"}'
```

```json
{
  "result": "Backpropagation is an algorithm used to train neural networks...",
  "score": 0.8923,
  "elapsed_ms": 51.3
}
```

**Generar embedding:**
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

**Métricas:**
```bash
curl http://localhost:8000/metrics
```

```json
{
  "total_requests": 5,
  "avg_embedding_ms": 46.8,
  "avg_search_ms": 3.1,
  "cache_hits": 12,
  "cache_misses": 3,
  "cache_hit_ratio": 0.8
}
```

**Documentación interactiva (Swagger UI):**
```
http://localhost:8000/docs
```

### 4. Correr el cliente Go

```bash
cd inference-client

# Con query por defecto
go run main.go

# Con query personalizada
go run main.go -query "explain the attention mechanism in transformers"

# Apuntando a otra URL
go run main.go -url http://localhost:8000 -query "what is a vector database?"
```

Salida esperada:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Inference Engineering — Go Client
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Service URL : http://localhost:8000
  Query       : explain the attention mechanism in transformers
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] Checking service health...
    Status             : ok
    Model loaded       : true
    Corpus initialized : true

[2] Sending search query...

    ┌─ Result ──────────────────────────────────────
    │  Attention mechanism allows neural networks to focus on relevant...
    └───────────────────────────────────────────────

    Similarity score   : 0.8734
    Service latency    : 52.10 ms
    Client RTT         : 54 ms

[3] Service metrics snapshot...
    Cache hit ratio    : 93.3%
```

### 5. Detener el servicio

```bash
docker-compose down          # detiene contenedores
docker-compose down -v       # detiene + elimina volúmenes (re-descarga el modelo)
```

---

## Endpoints API

| Método | Path | Descripción |
|---|---|---|
| `POST` | `/embedding` | Genera embedding de un texto (384 dims) |
| `POST` | `/search` | Busca el documento más similar en el corpus |
| `GET` | `/health` | Estado de readiness del servicio |
| `GET` | `/metrics` | Métricas de latencia y cache |
| `GET` | `/docs` | Swagger UI (OpenAPI) |

---

## Conceptos de Inference Engineering que aprenderás

### 1. Startup vs Request Time
El modelo se carga **una vez** en startup. Los embeddings del corpus se precomputan **una vez**. En runtime, el costo es mínimo. Este patrón es fundamental en ML serving.

### 2. Embedding Cache
Los embeddings son deterministas: el mismo texto siempre produce el mismo vector. El cache `dict[text → vector]` evita recomputar. En producción se usa Redis o una cache distribuida.

### 3. Cosine Similarity
La métrica fundamental de búsqueda semántica. `dot(A,B) / (||A|| * ||B||)`. Mide el ángulo entre vectores, no la distancia euclidiana. Independiente de la magnitud del vector.

### 4. Observabilidad
Sin métricas no hay producción. `/health` para readiness probes, `/metrics` para performance. Logs estructurados en JSON para sistemas de observabilidad (Datadog, CloudWatch).

### 5. Schema-First Design
Los contratos de datos (Pydantic schemas) se definen antes de la lógica. FastAPI valida automáticamente. Los consumidores del API conocen exactamente qué esperar.

### 6. Separación de Responsabilidades
- `models/`: contratos y modelo ML
- `services/`: lógica de negocio
- `utils/`: funciones puras reutilizables
- `main.py`: solo conecta piezas, no contiene lógica

### 7. Microservicios y contratos HTTP
El cliente Go consume el servicio Python vía HTTP. Deben modelar el mismo contrato de datos independientemente. Esto es la realidad de los microservicios.

---

## Posibles mejoras futuras

### Fase 2 — Vector Store
Reemplazar `data/documents.py` con ChromaDB. Agregar `POST /ingest` para cargar documentos dinámicamente. La interfaz de `search_service.py` no cambiaría.

### Fase 3 — RAG Completo
Agregar `POST /rag` que: busca contexto → construye prompt → llama a un LLM (Ollama) → retorna respuesta generada con fuentes.

### Fase 4 — Performance
- FAISS para búsqueda vectorial en O(log n) en lugar de O(n)
- Redis para cache de embeddings compartido entre workers
- Múltiples workers Uvicorn con modelo por proceso
- Métricas en formato Prometheus para Grafana

### Fase 5 — Producción
- Autenticación con API keys
- Rate limiting
- Monitoreo de latencia P50/P95/P99
- Despliegue en Railway, Fly.io o AWS ECS

---

## Stack tecnológico

| Componente | Tecnología | Versión |
|---|---|---|
| API Framework | FastAPI | 0.115.5 |
| Embeddings | sentence-transformers | 3.3.1 |
| Modelo ML | all-MiniLM-L6-v2 | — |
| Validación | Pydantic | 2.10.3 |
| ASGI Server | Uvicorn | 0.32.1 |
| Cliente | Go | 1.22 |
| Contenedor | Docker | >= 24.0 |
