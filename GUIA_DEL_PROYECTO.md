# 🚀 Guía Completa del Proyecto: Inference Service
## "El libro para devs web que quieren entender ML en producción"

> **Para quien es esto:** Si sabes hacer una API REST, has tocado Docker aunque sea de lejos, y oyes hablar de "embeddings" o "RAG" y piensas *"¿eso es una marca de ropa?"* — este libro es para ti. Bienvenido.

---

## 📋 Tabla de Contenidos

- [Capítulo 0 — ¿De qué va este proyecto?](#-capítulo-0-de-qué-va-este-proyecto)
- [Capítulo 1 — La arquitectura del sistema (el mapa del tesoro)](#️-capítulo-1-la-arquitectura-del-sistema-el-mapa-del-tesoro)
- [Capítulo 2 — La estructura de carpetas (¿dónde está todo?)](#-capítulo-2-la-estructura-de-carpetas-dónde-está-todo)
- [Capítulo 3 — Los conceptos clave que TIENES que entender](#-capítulo-3-los-conceptos-clave-que-tienes-que-entender)
- [Capítulo 4 — La API: Cómo hablar con el sistema](#-capítulo-4-la-api--cómo-hablar-con-el-sistema)
- [Capítulo 5 — El corazón del sistema: FastAPI y los servicios](#-capítulo-5-el-corazón-del-sistema--fastapi-y-los-servicios)
- [Capítulo 6 — Los datos: ChromaDB y Redis](#️-capítulo-6-los-datos--chromadb-y-redis)
- [Capítulo 7 — Docker y la infraestructura](#-capítulo-7-docker-y-la-infraestructura)
- [Capítulo 8 — Los tests: ¿Cómo sé que funciona?](#-capítulo-8-los-tests--cómo-sé-que-funciona)
- [Capítulo 9 — Observabilidad: Prometheus y Grafana](#-capítulo-9-observabilidad--prometheus-y-grafana)
- [Capítulo 10 — CI/CD: Automatizando la calidad](#-capítulo-10-cicd--automatizando-la-calidad)
- [Capítulo 11 — El cliente Go](#-capítulo-11-el-cliente-go)
- [Capítulo 12 — Guía de inicio rápido (TL;DR)](#-capítulo-12-guía-de-inicio-rápido-tldr)
- [Capítulo 13 — Flujo de datos completo: De la petición a la respuesta](#️-capítulo-13-flujo-de-datos-completo--de-la-petición-a-la-respuesta)
- [Capítulo 14 — Glosario: Las palabras raras explicadas](#-capítulo-14-glosario--las-palabras-raras-explicadas)
- [Capítulo 15 — ¿Qué aprendiste? Resumen final](#-capítulo-15-qué-aprendiste-resumen-final)

---

## 🎯 Capítulo 0: ¿De qué va este proyecto?

Imagínate que eres un dev web. Sabes montar APIs, conectar bases de datos, hacer deploys. Un día tu jefe llega y dice: *"Necesitamos integrar un modelo de Machine Learning en producción."* Y tú asientes con cara de "claro, claro" mientras por dentro piensas *"¿y eso cómo se come?"*.

Este proyecto existe exactamente para ese momento.

### 🤔 ¿Qué problema resuelve?

El proyecto implementa un **sistema de inferencia** — es decir, un sistema que toma un modelo de ML ya entrenado y lo expone como un servicio web que cualquier aplicación puede consumir. Específicamente hace tres cosas:

1. **Convierte texto en vectores numéricos** (embeddings): Transforma frases como "¿cómo funciona el aprendizaje profundo?" en una lista de 384 números que capturan el *significado* de esa frase.

2. **Busca documentos por significado** (búsqueda semántica): No busca palabras exactas como Google de los 90, sino que encuentra documentos que *significan cosas similares*, aunque usen palabras distintas.

3. **Responde preguntas usando contexto** (RAG): Antes de pedirle a un LLM (como ChatGPT, pero local) que responda una pregunta, primero busca los documentos más relevantes y se los da como contexto. Así el modelo no alucina tanto.

### 👤 ¿Para quién es?

- Devs web que quieren entender cómo funciona ML en producción sin un doctorado en matemáticas
- Personas que han oído hablar de RAG, embeddings, vectores, y quieren ver código real
- Estudiantes de Ingeniería de Inferencia (el arte de desplegar modelos de ML)

### 🎓 ¿Qué aprenderás?

Al terminar esta guía entenderás:

- Cómo funciona un sistema de búsqueda semántica de principio a fin
- Qué es RAG y por qué todos hablan de él
- Cómo se construye un sistema de caché de dos niveles
- Cómo monitorizar un servicio de ML con Prometheus y Grafana
- Cómo orquestar múltiples servicios con Docker Compose
- Cómo una API en Python puede hablar con un cliente en Go

### 🍕 La analogía del sistema completo

Piensa en este sistema como una **pizzería con sistema de delivery muy sofisticado**:

- El **cliente** hace un pedido (tu petición HTTP)
- El **recepcionista** verifica que el cliente tiene cuenta (autenticación)
- El **supervisor de turno** controla que nadie haga demasiados pedidos seguidos (rate limiting)
- El **almacén de ingredientes** guarda los documentos que el sistema conoce (ChromaDB)
- El **registro de clientes frecuentes** recuerda los pedidos recientes para no repetir trabajo (Redis cache)
- El **cocinero especialista** convierte los textos en vectores (modelo de embeddings)
- El **chef ejecutivo** prepara la respuesta final con toda la información (LLM via Ollama/vLLM)
- El **panel de control de la cocina** muestra métricas de rendimiento (Prometheus + Grafana)

Todo coordinado, todo automatizado, todo listo para producción.

---

## 🗺️ Capítulo 1: La arquitectura del sistema (el mapa del tesoro)

Antes de tocar una sola línea de código, necesitas ver el mapa completo. Sin mapa, te pierdes en el bosque de archivos y no entiendes por qué existe cada cosa.

### 🏗️ Diagrama de la arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                        MUNDO EXTERIOR                           │
│                                                                 │
│   Tu app / curl / navegador / cliente Go                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP Request
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   inference-service (Python/FastAPI)            │
│                   Puerto 8000                                   │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │    Auth     │──▶│ Rate Limiter │──▶│      Routes          │ │
│  │  (Bearer)   │   │  (Sliding    │   │  /embedding          │ │
│  └─────────────┘   │   Window)    │   │  /search             │ │
│                    └──────────────┘   │  /ingest             │ │
│                                       │  /rag                │ │
│                                       │  /health             │ │
│                                       │  /metrics            │ │
│                                       └──────────┬───────────┘ │
│                                                  │             │
│  ┌──────────────────────────────────────────┐    │             │
│  │           Servicios Internos             │◀───┘             │
│  │                                          │                  │
│  │  EmbeddingService ──▶ L1 Cache (dict)    │                  │
│  │         │           ──▶ L2 Cache (Redis) │                  │
│  │         │           ──▶ Modelo ML        │                  │
│  │         ▼                                │                  │
│  │  SearchService ──▶ ChromaDB (HNSW)       │                  │
│  │         │                                │                  │
│  │         ▼                                │                  │
│  │  RAGService ──▶ Ollama/vLLM (LLM)        │                  │
│  └──────────────────────────────────────────┘                  │
└─────────┬────────────────────────────────────────┬─────────────┘
          │                                        │
          ▼                                        ▼
┌─────────────────────┐                ┌──────────────────────────┐
│    Redis (Puerto     │                │  ChromaDB (en disco)      │
│    6379)             │                │  Vector Store + HNSW      │
│  - Caché embeddings  │                │  - 15+ documentos         │
│  - Rate limiting     │                │  - Índice vectorial       │
└─────────────────────┘                └──────────────────────────┘
          
┌─────────────────────┐                ┌──────────────────────────┐
│  Ollama (Puerto      │                │  Observabilidad           │
│  11434)              │                │                          │
│  - LLM local         │                │  Prometheus (9090)        │
│  - llama3.2:1b       │                │  Grafana (3000)           │
└─────────────────────┘                └──────────────────────────┘
```

### 🎭 ¿Qué hace cada servicio en lenguaje humano?

| Servicio | Tecnología | Puerto | Para qué sirve |
|---|---|---|---|
| **inference-service** | Python + FastAPI | 8000 | El cerebro. Procesa todas las peticiones |
| **Redis** | Redis 7 | 6379 | La memoria rápida. Caché y control de tráfico |
| **ChromaDB** | ChromaDB | (interno) | La biblioteca. Guarda y busca documentos por significado |
| **Ollama** | Ollama | 11434 | El escritor. Genera respuestas de texto con un LLM local |
| **Prometheus** | Prometheus | 9090 | El contador. Recoge métricas del sistema |
| **Grafana** | Grafana | 3000 | El panel de control. Visualiza las métricas |

### 🔄 ¿Cómo se comunican?

Todos los servicios viven en la misma **red Docker virtual** llamada `inference-net`. Es como si estuvieran en el mismo edificio de oficinas — pueden llamarse entre ellos por nombre (no por IP), y el mundo exterior solo puede entrar por las puertas que se abren explícitamente.

- El `inference-service` habla con Redis via **cliente Redis** (protocolo propio, muy rápido)
- El `inference-service` habla con Ollama via **HTTP** (REST API)
- El `inference-service` expone métricas y Prometheus las **scrape** periódicamente vía HTTP
- Grafana lee datos de Prometheus via **HTTP**
- Tú hablas con `inference-service` via **HTTP** en el puerto 8000

> [!NOTE]
> ChromaDB en este proyecto se usa como **librería embebida**, no como servicio separado. Vive dentro del proceso Python y guarda los datos en disco (un volumen Docker). No tiene puerto propio.

---

## 📁 Capítulo 2: La estructura de carpetas (¿dónde está todo?)

Navegar un proyecto nuevo es como entrar a una casa desconocida en la oscuridad. Necesitas saber dónde está el interruptor de la luz antes de empezar a moverte. Este capítulo es ese interruptor.

### 🏠 La analogía de la casa

Piensa en el proyecto como una casa con habitaciones especializadas:

```
proyecto/                       ← La casa entera
│
├── 📋 docker-compose.yml       ← El plano arquitectónico (cómo encaja todo)
├── 📋 .env.example             ← El manual de la calefacción (variables de entorno)
├── 📋 README.md                ← El cartel en la puerta (instrucciones básicas)
│
├── 🐍 inference-service/       ← LA HABITACIÓN PRINCIPAL (el backend)
│   │
│   ├── 🚪 main.py              ← La puerta de entrada (rutas y configuración)
│   │
│   ├── 🛠️ services/            ← Los electrodomésticos (la lógica de negocio)
│   │   ├── auth.py             ← El portero (autenticación)
│   │   ├── rate_limiter.py     ← El guardia de seguridad (control de tráfico)
│   │   ├── embedding_service.py← El traductor (texto → vectores)
│   │   ├── search_service.py   ← El buscador (encontrar documentos similares)
│   │   ├── vector_store.py     ← El archivero (gestión de ChromaDB)
│   │   ├── rag_service.py      ← El investigador (RAG pipeline)
│   │   ├── ollama_client.py    ← El teléfono de Ollama
│   │   ├── vllm_client.py      ← El teléfono de vLLM
│   │   ├── redis_cache.py      ← El post-it gigante (caché Redis)
│   │   ├── prometheus_metrics.py← El cuaderno de contabilidad
│   │   ├── health_service.py   ← El termómetro (¿está el sistema sano?)
│   │   └── metrics_store.py    ← El cajón de estadísticas
│   │
│   ├── 📐 models/              ← Los moldes (definiciones de datos)
│   │   ├── embedding_model.py  ← El modelo ML (SentenceTransformer)
│   │   └── schemas.py          ← Los formularios (qué acepta y devuelve la API)
│   │
│   ├── 🔧 utils/               ← Las herramientas del cajón
│   │   ├── similarity.py       ← La calculadora de parecido
│   │   ├── timer.py            ← El cronómetro
│   │   └── logger.py           ← El diario (logs estructurados)
│   │
│   ├── 📚 data/
│   │   └── documents.py        ← Los 15 documentos de ejemplo pre-cargados
│   │
│   └── 🧪 tests/               ← El laboratorio de control de calidad
│       ├── test_similarity.py
│       ├── test_schemas.py
│       ├── test_timer.py
│       └── test_metrics_store.py
│
├── 🐹 inference-client/         ← La habitación del cliente (CLI en Go)
│   ├── go.mod
│   └── main.go
│
└── 📊 observability/            ← La sala de control (monitorización)
    ├── prometheus/prometheus.yml
    └── grafana/
        ├── dashboards/
        └── provisioning/
```

### 📍 ¿Por dónde empezar si eres nuevo en el proyecto?

Sigue este orden de lectura recomendado:

1. **`docker-compose.yml`** — Para entender qué servicios existen
2. **`.env.example`** — Para saber qué configuraciones hay disponibles
3. **`inference-service/main.py`** — La puerta de entrada, donde se definen las rutas
4. **`inference-service/models/schemas.py`** — Qué datos entran y salen de la API
5. **`inference-service/services/embedding_service.py`** — El servicio más importante
6. **`inference-service/services/rag_service.py`** — La pieza más compleja

> [!TIP]
> Antes de leer el código de servicios, lee siempre `schemas.py` primero. Entender la forma de los datos que maneja cada servicio te da el 50% del contexto que necesitas.

---

## 🧠 Capítulo 3: Los conceptos clave que TIENES que entender

Este es el capítulo más importante del libro. Si entiendes estos conceptos, el resto del proyecto se vuelve obvio. Si no los entiendes, estarás mirando código sin saber para qué sirve.

### 🎨 Embeddings: Convirtiendo texto en números

Imagina que tienes que describir colores a alguien que no puede verlos, pero sí puede medir cosas. Podrías decir:
- Rojo: `(255, 0, 0)` — mucho rojo, nada de verde, nada de azul
- Azul marino: `(0, 0, 128)` — nada de rojo, nada de verde, algo de azul
- Morado: `(128, 0, 128)` — algo de rojo, nada de verde, algo de azul

Los colores que son **similares** tienen números **similares**. El rosa `(255, 192, 203)` está cerca del rojo `(255, 0, 0)` en ese espacio numérico.

Los **embeddings** hacen exactamente esto, pero con texto y con 384 dimensiones en lugar de 3.

```
"aprendizaje automático"  →  [0.23, -0.45, 0.12, ..., 0.67]  (384 números)
"machine learning"        →  [0.24, -0.44, 0.11, ..., 0.65]  (384 números)
"recetas de cocina"       →  [-0.89, 0.34, -0.23, ..., 0.01] (384 números)
```

¿Ves? "aprendizaje automático" y "machine learning" tienen números muy parecidos porque **significan lo mismo**. "recetas de cocina" tiene números completamente distintos porque **significa algo diferente**.

El modelo que hace esta magia en este proyecto se llama **`all-MiniLM-L6-v2`**. Es un modelo pequeño (22 millones de parámetros, ~90MB) pero muy eficiente. Produce vectores de **384 dimensiones**.

> [!NOTE]
> ¿Por qué 384 dimensiones? Es un equilibrio entre calidad de representación y velocidad. Modelos más grandes usan 768 o 1536 dimensiones, pero son más lentos. Para búsqueda semántica general, 384 es suficiente.

### 🔍 Búsqueda Vectorial / Búsqueda Semántica

Si los textos son puntos en un espacio de 384 dimensiones, **buscar documentos similares** es simplemente encontrar los puntos más cercanos a tu pregunta.

Imagina una biblioteca donde cada libro está colocado físicamente según su contenido — los libros de cocina están juntos, los de ciencia ficción juntos, los de historia juntos. Si buscas "novelas de viajes en el tiempo", vas al área de ciencia ficción y buscas en los estantes cercanos, aunque ningún libro tenga exactamente esas palabras en el título.

Eso es la búsqueda semántica.

**¿Qué es HNSW?** El algoritmo que usa ChromaDB para buscar eficientemente se llama *Hierarchical Navigable Small World*. Imagínalo como un sistema de autopistas:

```
Nivel 3 (autopista):    A -------- E -------- I
                        |                     |
Nivel 2 (carretera):    A --- C --- E --- G --- I
                        |    |     |    |     |
Nivel 1 (calle):        A-B--C-D--E-F--G-H--I-J
```

En lugar de comparar tu búsqueda con TODOS los documentos (lento para millones de docs), HNSW navega por niveles: primero encuentra una zona aproximada en el nivel alto, luego refina la búsqueda en niveles más detallados. Es muchísimo más rápido.

### 📖 RAG: El estudiante con libros de texto

**RAG** significa *Retrieval-Augmented Generation* (Generación Aumentada por Recuperación). El nombre es técnico, pero la idea es simple.

Imagina que eres un estudiante con un examen de historia. Tienes dos opciones:
1. Responder solo de memoria (como lo hace un LLM normal) — puedes alucinar fechas o eventos
2. Tener tus libros de texto delante, buscar la respuesta, y luego redactarla con tus palabras (RAG)

La segunda opción es muchísimo más fiable. Eso es exactamente lo que hace RAG:

```
Pregunta del usuario
        │
        ▼
1. RETRIEVE: Busca los 3 documentos más relevantes en ChromaDB
        │
        ▼
2. AUGMENT: Construye un prompt con la pregunta + los documentos encontrados:
   "Contexto:
    [Doc 1]: Los transformers son arquitecturas de redes neuronales...
    [Doc 2]: El mecanismo de atención permite...
    
    Pregunta: ¿Cómo funcionan los transformers?
    Responde basándote en el contexto anterior."
        │
        ▼
3. GENERATE: El LLM (Ollama/vLLM) genera una respuesta usando ese contexto
        │
        ▼
Respuesta fundamentada en documentos reales
```

El resultado es una respuesta mucho más precisa y menos propensa a inventarse cosas.

### ⚡ La caché de dos niveles (L1 + L2 + L3)

Calcular un embedding no es gratis. El modelo tarda ~50ms por texto. Si recibes 1000 peticiones por segundo con los mismos textos, estarías desperdiciando tiempo. La solución: **caché**.

Este proyecto tiene tres niveles, como las cajas registradoras en un supermercado grande:

```
┌─────────────────────────────────────────────────────┐
│                  PETICIÓN ENTRANTE                  │
│                 "¿Qué es un vector?"                │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────┐
│  L1: Caché en memoria del proceso │  ← Sub-microsegundo
│  (diccionario Python en RAM)      │     ¿Lo tengo yo?
│  Rápidísimo, pero solo local      │
└───────────────────────┬───────────┘
                        │ MISS (no está)
                        ▼
┌───────────────────────────────────┐
│  L2: Redis (caché compartida)     │  ← ~0.5 milisegundos
│  Compartida entre todos los       │     ¿Lo tiene alguien?
│  workers, 24h de vida             │
└───────────────────────┬───────────┘
                        │ MISS (no está)
                        ▼
┌───────────────────────────────────┐
│  L3: Modelo ML                    │  ← ~50 milisegundos
│  SentenceTransformer              │     Calcularlo de cero
│  El más lento, la fuente de verdad│
└───────────────────────────────────┘
```

La analogía del supermercado:
- **L1** es el bolsillo de tu chaqueta — instantáneo, pero solo lo que cabe ahí
- **L2** es el maletero del coche — tardas 30 segundos en ir, pero cabe más y lo puede usar más gente
- **L3** es el supermercado — tienes todo, pero tardas 20 minutos en ir

Para la misma consulta, el segundo acceso es 100x más rápido porque ya está en caché.

### 🚦 Rate Limiting: El portero del club

El rate limiting es el sistema que evita que un solo cliente sature el servidor haciendo demasiadas peticiones.

Imagina que tienes una fuente de agua pública. Si una sola persona viene con una manguera industrial y vacía la fuente, el resto no puede beber. El portero del club (rate limiter) dice: "tú, solo 60 vasos por minuto".

Este proyecto usa un algoritmo de **ventana deslizante** (*Sliding Window*):

```
Tiempo:  [--- 60 segundos deslizantes ---]
         
IP: 192.168.1.100

Petición 1  ──── [✅ cuenta: 1/60]
Petición 2  ──── [✅ cuenta: 2/60]
...
Petición 60 ──── [✅ cuenta: 60/60]
Petición 61 ──── [❌ BLOQUEADO - 429 Too Many Requests]

(60 segundos después de la petición 1...)

Petición 62 ──── [✅ cuenta: 60/60]  ← La petición 1 ya "expiró"
```

La ventana deslizante es más justa que la ventana fija (que resetea de golpe cada minuto) porque no permite que alguien haga 60 peticiones al final de un minuto y 60 más al inicio del siguiente, consiguiendo 120 en 2 segundos.

---

## 🔌 Capítulo 4: La API — Cómo hablar con el sistema

Una API es el contrato entre tu cliente y el servidor. Este sistema expone 8 endpoints. Vamos a ver cada uno con ejemplos reales que puedes ejecutar en tu terminal.

### 🔑 Autenticación con Bearer Token

Antes de ver los endpoints, necesitas saber cómo autenticarte. La mayoría de endpoints requieren un **Bearer Token** — básicamente una contraseña que mandas en la cabecera HTTP.

Si tienes configurada la variable `API_KEYS=mi-clave-secreta`, todas tus peticiones deben incluir:

```
Authorization: Bearer mi-clave-secreta
```

Si `API_KEYS` está vacía, la autenticación está desactivada y puedes hacer peticiones sin cabecera.

> [!WARNING]
> En producción NUNCA dejes `API_KEYS` vacío. Cualquiera podría usar tu API y consumir tus recursos (y probablemente tu factura de servidor).

### 📍 Endpoint 1: GET /health — "¿Estás vivo?"

El endpoint más simple. No requiere autenticación. Sirve para saber si el servicio está listo para recibir peticiones.

```bash
curl http://localhost:8000/health
```

Respuesta:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "chromadb_ready": true,
  "redis_connected": true
}
```

Si `model_loaded` es `false`, significa que el SentenceTransformer todavía está cargando (puede tardar unos segundos al arrancar). Hasta que no sea `true`, el resto de endpoints no funcionarán bien.

### 📍 Endpoint 2: POST /embedding — "Convierte esto en números"

Toma un texto y devuelve su vector de 384 dimensiones.

```bash
curl -X POST http://localhost:8000/embedding \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer tu-api-key" \
  -d '{"text": "¿Qué es el aprendizaje profundo?"}'
```

Respuesta (truncada para legibilidad):
```json
{
  "embedding": [0.0234, -0.4521, 0.1123, ..., 0.0678],
  "dimensions": 384,
  "cached": false,
  "latency_ms": 47.3
}
```

El campo `cached` te dice si vino de la caché (rápido) o del modelo (más lento). La primera vez es `false`, la segunda vez es `true` y el `latency_ms` baja dramáticamente.

### 📍 Endpoint 3: POST /search — "Encuentra documentos similares"

El corazón de la búsqueda semántica. Toma una pregunta y devuelve los documentos más similares del almacén.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer tu-api-key" \
  -d '{
    "query": "cómo funcionan las redes neuronales",
    "top_k": 3
  }'
```

Respuesta:
```json
{
  "results": [
    {
      "id": "doc_001",
      "text": "Las redes neuronales artificiales son sistemas computacionales...",
      "score": 0.892,
      "metadata": {"category": "deep_learning"}
    },
    {
      "id": "doc_007",
      "text": "El perceptrón multicapa es la arquitectura básica de...",
      "score": 0.834,
      "metadata": {"category": "neural_networks"}
    },
    {
      "id": "doc_012",
      "text": "El backpropagation es el algoritmo que permite entrenar...",
      "score": 0.801,
      "metadata": {"category": "training"}
    }
  ],
  "query_embedding_latency_ms": 2.1,
  "search_latency_ms": 0.8
}
```

El campo `score` va de 0 a 1, donde 1 significa idéntico. Todo lo que supere 0.7 suele ser muy relevante.

> [!TIP]
> El parámetro `top_k` acepta valores de 1 a 20. Empieza con 3–5 para RAG, y hasta 10–20 si quieres mostrar una lista de resultados al usuario.

### 📍 Endpoint 4: POST /ingest — "Aprende este documento"

Agrega un nuevo documento al almacén de vectores para que aparezca en búsquedas futuras.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer tu-api-key" \
  -d '{
    "id": "mi_doc_001",
    "text": "Los transformers revolucionaron el procesamiento de lenguaje natural...",
    "metadata": {
      "source": "mi_blog",
      "category": "NLP"
    }
  }'
```

Respuesta:
```json
{
  "status": "upserted",
  "id": "mi_doc_001",
  "latency_ms": 51.2
}
```

"Upserted" significa que si ya existía un documento con ese ID, lo actualiza; si no existía, lo crea. Esta es la operación más lenta porque requiere calcular el embedding Y guardarlo en ChromaDB.

### 📍 Endpoint 5: POST /rag — "Responde esta pregunta con contexto"

El endpoint estrella. Combina búsqueda semántica con generación de texto.

```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer tu-api-key" \
  -d '{
    "question": "¿Cuál es la diferencia entre supervised y unsupervised learning?",
    "top_k": 3
  }'
```

Respuesta:
```json
{
  "answer": "El aprendizaje supervisado utiliza datos etiquetados para entrenar modelos, donde cada ejemplo tiene una respuesta correcta conocida. El aprendizaje no supervisado, en cambio, trabaja con datos sin etiquetas, buscando patrones y estructuras inherentes...",
  "sources": [
    {"id": "doc_003", "score": 0.89, "text": "El aprendizaje supervisado..."},
    {"id": "doc_008", "score": 0.82, "text": "Los algoritmos de clustering..."},
    {"id": "doc_011", "score": 0.78, "text": "K-means y PCA son..."}
  ],
  "retrieval_latency_ms": 3.2,
  "generation_latency_ms": 1240.5,
  "total_latency_ms": 1245.1
}
```

Nota que `generation_latency_ms` es mucho mayor — ahí es donde el LLM genera texto, lo más lento del pipeline.

### 📍 Endpoint 6: GET /metrics — "¿Cómo va el sistema?"

Devuelve estadísticas del servicio en JSON legible por humanos.

```bash
curl http://localhost:8000/metrics
```

Respuesta:
```json
{
  "total_requests": 1523,
  "embedding_cache_hits_l1": 892,
  "embedding_cache_hits_l2": 445,
  "embedding_cache_misses": 186,
  "avg_embedding_latency_ms": 2.3,
  "avg_search_latency_ms": 1.1,
  "avg_rag_latency_ms": 1340.2,
  "rate_limit_blocks": 12
}
```

### 📍 Endpoint 7: GET /metrics/prometheus — "Métricas para máquinas"

El mismo tipo de información, pero en formato Prometheus (texto plano con etiquetas). Prometheus lo scrapea automáticamente.

```bash
curl http://localhost:8000/metrics/prometheus
```

```
# HELP inference_requests_total Total number of requests
# TYPE inference_requests_total counter
inference_requests_total{endpoint="/embedding"} 523
inference_requests_total{endpoint="/search"} 412
inference_requests_total{endpoint="/rag"} 588
# HELP embedding_latency_seconds Embedding generation latency
# TYPE embedding_latency_seconds histogram
embedding_latency_seconds_bucket{le="0.01"} 892
...
```

### 📍 Endpoint 8: GET /docs — "El manual interactivo"

FastAPI genera automáticamente una interfaz Swagger. Abre en el navegador:

```
http://localhost:8000/docs
```

Desde ahí puedes probar todos los endpoints visualmente, sin necesidad de curl.

---

## 🐍 Capítulo 5: El corazón del sistema — FastAPI y los servicios

Si el proyecto fuera un coche, `main.py` sería el cuadro de mandos y los archivos en `services/` serían el motor, la caja de cambios, y los frenos. Este capítulo te explica cómo funcionan por dentro.

### 🚪 `main.py` — La puerta de entrada

Este archivo hace tres cosas fundamentales:

**1. Arranque del sistema (lifespan):**

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP — esto corre ANTES de aceptar peticiones
    await load_sentence_transformer()   # Carga el modelo ML (~5 segundos)
    await init_redis()                  # Conecta con Redis
    await init_chromadb()               # Inicializa la base de datos vectorial
    await precompute_seed_corpus()      # Ingesta los 15 documentos iniciales
    await verify_ollama_model()         # Verifica que el LLM está disponible
    
    yield  # ← El servicio está listo para recibir peticiones
    
    # SHUTDOWN — esto corre al apagar
    await cleanup_resources()
```

El patrón `lifespan` de FastAPI es moderno y elegante. Todo lo que va antes del `yield` es setup; todo lo que va después es teardown. Es como encender/apagar el local de una tienda.

**2. Middleware (lo que ocurre en TODAS las peticiones):**

```python
# Orden de ejecución (de afuera hacia adentro):
# Petición → CORS → Auth → Rate Limit → Tu Handler → Respuesta
app.add_middleware(CORSMiddleware, ...)
app.add_middleware(AuthMiddleware, ...)
app.add_middleware(RateLimitMiddleware, ...)
```

Los middlewares son como los filtros de un embudo. Cada petición pasa por todos ellos antes de llegar al código del endpoint.

**3. Las rutas (qué URL hace qué):**

```python
@app.post("/embedding")
async def create_embedding(request: EmbeddingRequest):
    ...

@app.post("/search")
async def semantic_search(request: SearchRequest):
    ...
```

### 🔐 `services/auth.py` — El portero

Simple pero efectivo. Lee la cabecera `Authorization: Bearer <token>` y verifica si el token está en la lista de `API_KEYS` (variable de entorno).

```python
async def verify_token(credentials: HTTPAuthorizationCredentials):
    if not API_KEYS:  # Si no hay claves configuradas, se deja pasar todo
        return True
    if credentials.credentials not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid token")
    return True
```

> [!NOTE]
> Esto es autenticación, no autorización. La autenticación responde "¿eres quien dices ser?". La autorización respondería "¿puedes hacer lo que quieres hacer?". Este sistema solo hace lo primero — todos los tokens válidos pueden hacer todo.

### 🚦 `services/rate_limiter.py` — El guardia de seguridad

Implementa el algoritmo de ventana deslizante usando un **Redis Sorted Set (ZSET)**. La clave es por IP del cliente:

```python
# Para cada petición de la IP "192.168.1.100":
async def check_rate_limit(ip: str) -> bool:
    now = time.time()
    window_start = now - WINDOW_SECONDS  # Hace 60 segundos
    key = f"ratelimit:{ip}"
    
    async with redis.pipeline() as pipe:
        # 1. Eliminar peticiones fuera de la ventana
        pipe.zremrangebyscore(key, 0, window_start)
        # 2. Contar las que quedan
        pipe.zcard(key)
        # 3. Añadir la petición actual
        pipe.zadd(key, {str(now): now})
        # 4. Establecer expiración del key
        pipe.expire(key, WINDOW_SECONDS)
        results = await pipe.execute()
    
    count = results[1]
    return count <= MAX_REQUESTS  # True = permitir, False = bloquear
```

> [!TIP]
> El sistema es **fail-open**: si Redis no está disponible, el rate limiting se salta y la petición pasa. Esto evita que un problema de Redis tumbe toda tu API, a costa de un posible abuso durante ese período.

### 🧠 `services/embedding_service.py` — El traductor

El servicio más importante del sistema. Orquesta los tres niveles de caché:

```python
async def get_embedding(text: str) -> list[float]:
    cache_key = hash(text)  # Simplificado para el ejemplo
    
    # L1: ¿Está en memoria local?
    if cache_key in L1_CACHE:
        return L1_CACHE[cache_key]  # < 1 microsegundo
    
    # L2: ¿Está en Redis?
    cached = await redis_cache.get(cache_key)
    if cached:
        L1_CACHE[cache_key] = cached  # Lo guardo en L1 para la próxima
        return cached  # ~0.5ms
    
    # L3: Calcular con el modelo
    embedding = model.encode(text)  # ~50ms
    
    # Guardar en ambas cachés para futuras peticiones
    L1_CACHE[cache_key] = embedding
    await redis_cache.set(cache_key, embedding, ttl=86400)
    
    return embedding
```

### 🔍 `services/search_service.py` — El buscador

Recibe una consulta, la convierte en embedding, y busca en ChromaDB:

```python
async def search(query: str, top_k: int = 5) -> list[SearchResult]:
    # 1. Convertir la pregunta en vector
    query_embedding = await embedding_service.get_embedding(query)
    
    # 2. Buscar en ChromaDB (HNSW internamente)
    results = await vector_store.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # 3. Formatear y devolver
    return format_results(results)
```

### 🤖 `services/rag_service.py` — El investigador

El pipeline completo de RAG. También implementa el **patrón Factory** para elegir el backend LLM:

```python
def get_llm_client(backend: str):
    """Factory Pattern: devuelve el cliente correcto según la configuración"""
    if backend == "ollama":
        return OllamaClient(base_url=OLLAMA_BASE_URL)
    elif backend == "vllm":
        return VLLMClient(base_url=VLLM_BASE_URL)
    else:
        raise ValueError(f"Backend desconocido: {backend}")

async def answer_question(question: str, top_k: int = 3) -> RAGResponse:
    # 1. RETRIEVE
    search_results = await search_service.search(question, top_k=top_k)
    
    # 2. AUGMENT — construir el prompt con contexto
    context = "\n".join([
        f"[{i+1}] (relevancia: {r.score:.2f}): {r.text}"
        for i, r in enumerate(search_results)
    ])
    
    prompt = f"""Basándote únicamente en el siguiente contexto, responde la pregunta.
    
Contexto:
{context}

Pregunta: {question}

Respuesta:"""
    
    # 3. GENERATE
    llm_client = get_llm_client(INFERENCE_BACKEND)
    answer = await llm_client.generate(
        prompt=prompt,
        temperature=0.3,
        max_tokens=512
    )
    
    return RAGResponse(answer=answer, sources=search_results)
```

El **patrón Factory** es elegante: puedes cambiar de Ollama a vLLM simplemente cambiando una variable de entorno (`INFERENCE_BACKEND=vllm`). El código del pipeline RAG no cambia nada.

---

## 🗄️ Capítulo 6: Los datos — ChromaDB y Redis

Detrás de todo sistema hay datos. En este proyecto hay dos tipos de almacenamiento muy diferentes, cada uno optimizado para lo que hace. Este capítulo explica qué guarda cada uno y por qué.

### 📚 ChromaDB — La biblioteca vectorial

ChromaDB es una base de datos especializada en vectores. No es como PostgreSQL o MongoDB, donde guardas registros estructurados. ChromaDB está optimizada para una sola pregunta: **"¿qué vectores son más similares a este?"**

**¿Qué guarda ChromaDB en este proyecto?**

Cada documento ingresado se guarda con tres partes:

```
┌─────────────────────────────────────────────────────┐
│                    DOCUMENTO                        │
├─────────────────┬───────────────┬───────────────────┤
│       ID        │     TEXTO     │     VECTOR        │
│   "doc_001"     │  "Las redes   │  [0.23, -0.45,    │
│                 │  neuronales   │   0.12, ...,       │
│                 │  son..."      │   0.67]  (384 dim) │
├─────────────────┴───────────────┴───────────────────┤
│                   METADATA                          │
│  {"category": "deep_learning", "source": "manual"}  │
└─────────────────────────────────────────────────────┘
```

**¿Cómo funciona HNSW internamente?** (Simplificado)

HNSW construye un grafo donde los nodos son documentos y las aristas conectan documentos similares. Cuando buscas, navegas por el grafo eligiendo siempre el vecino más cercano a tu consulta, hasta que llegas a los mejores resultados. Es como navegar por un mapa de relaciones.

**Los 15 documentos semilla**

En `data/documents.py` hay 15 documentos sobre temas de ML/IA que se cargan automáticamente al arrancar el sistema. Son el corpus inicial que permite que las búsquedas funcionen desde el primer momento, sin necesidad de ingestar nada. Cubren temas como:

- Tipos de aprendizaje (supervisado, no supervisado, por refuerzo)
- Algoritmos clásicos (regresión, clasificación, clustering)
- Deep learning y arquitecturas de redes
- Procesamiento de lenguaje natural
- Evaluación de modelos

### 🔴 Redis — La memoria ultrarrápida

Redis es una base de datos en memoria (*in-memory*). Todo vive en RAM, lo que la hace extremadamente rápida pero volátil (si reinicias, los datos pueden perderse sin configuración extra).

En este proyecto Redis tiene **dos responsabilidades**:

**1. Caché de embeddings (L2)**

Los embeddings se serializan en binario y se guardan con TTL de 24 horas:

```
Clave:  "emb:<hash_del_texto>"
Valor:  <bytes binarios del vector float32>
TTL:    86400 segundos (24 horas)
```

¿Por qué binario? Un vector de 384 floats en JSON tendría ~5KB. En binario es ~1.5KB. Menos datos = más velocidad y menos memoria.

**2. Control de rate limiting (ZSET)**

```
Clave:  "ratelimit:<ip_del_cliente>"
Valor:  Sorted Set con timestamps de cada petición
TTL:    60 segundos (la ventana de rate limit)
```

El Sorted Set es perfecto para esto: el "score" de cada elemento es el timestamp de la petición, y puedes eficientemente eliminar todos los elementos con score menor que `ahora - 60s`.

> [!NOTE]
> Redis persiste sus datos de forma opcional. En este proyecto, si reinicias el contenedor de Redis, el caché se vacía. Eso está bien — el caché se reconstruye automáticamente con el uso. El rate limiting también se reinicia, lo cual también es aceptable.

---

## 🐳 Capítulo 7: Docker y la infraestructura

Antes de Docker, desplegar una aplicación era como preparar una mudanza donde cada habitación tenía reglas de altura, temperatura y humedad distintas. Docker resuelve eso poniendo cada habitación en su propio contenedor con su propio clima controlado.

### 🎭 ¿Qué hace Docker Compose?

`docker-compose.yml` es un archivo YAML que describe **6 servicios** y cómo se relacionan. Con un solo comando (`docker compose up`) levanta todos los servicios en el orden correcto, con las variables de entorno, los volúmenes y las redes configuradas.

### 📦 Los 6 servicios en detalle

**1. inference-service**
```yaml
inference-service:
  build: ./inference-service          # Construye la imagen desde el Dockerfile
  ports: ["8000:8000"]               # Puerto 8000 del host → puerto 8000 del contenedor
  environment:
    - REDIS_URL=redis://redis:6379   # Nota: usa "redis" (nombre del servicio), no localhost
    - OLLAMA_BASE_URL=http://ollama:11434
    - WORKERS=2                      # 2 workers de Uvicorn
  depends_on: [redis, ollama]        # No arranca hasta que redis y ollama estén listos
  volumes:
    - chromadb_data:/app/chromadb    # Persistencia de la BD vectorial
    - hf_cache:/app/.cache           # Persistencia del modelo descargado
```

**2. redis**
```yaml
redis:
  image: redis:7-alpine   # Alpine = imagen mínima, más pequeña y segura
  ports: ["6379:6379"]
  volumes:
    - redis_data:/data    # Persiste los datos entre reinicios (opcional)
```

**3. ollama**
```yaml
ollama:
  image: ollama/ollama
  ports: ["11434:11434"]
  volumes:
    - ollama_models:/root/.ollama   # Los modelos descargados persisten
```

**4. prometheus**
```yaml
prometheus:
  image: prom/prometheus
  ports: ["9090:9090"]
  volumes:
    - ./observability/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
```

**5. grafana**
```yaml
grafana:
  image: grafana/grafana
  ports: ["3000:3000"]
  environment:
    - GF_SECURITY_ADMIN_USER=${GRAFANA_USER:-admin}
    - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
  volumes:
    - ./observability/grafana:/etc/grafana/provisioning
    - grafana_data:/var/lib/grafana
```

### 💾 Volúmenes y por qué importan

Sin volúmenes, cada vez que reinicias un contenedor, pierdes todos los datos que generó. Es como un ordenador sin disco duro — arranca, trabaja, y al apagarlo todo desaparece.

Los volúmenes en este proyecto:

| Volumen | Qué guarda | ¿Por qué importa? |
|---|---|---|
| `chromadb_data` | Los documentos vectorizados | Sin esto, pierdes todo el corpus al reiniciar |
| `hf_cache` | El modelo de embeddings descargado | Sin esto, re-descarga ~90MB en cada inicio |
| `ollama_models` | El modelo LLM (llama3.2:1b, ~1GB) | Sin esto, re-descarga ~1GB en cada inicio |
| `redis_data` | Caché de embeddings | Se reconstruye, pero es más rápido conservarlo |
| `grafana_data` | Dashboards y configuración de Grafana | Sin esto, pierdes tus dashboards personalizados |

### 🚀 Comandos esenciales de Docker Compose

```bash
# Iniciar todos los servicios (en segundo plano)
docker compose up -d

# Ver los logs en tiempo real
docker compose logs -f

# Ver los logs de un servicio específico
docker compose logs -f inference-service

# Parar todo
docker compose down

# Parar y eliminar volúmenes (¡cuidado! borra los datos)
docker compose down -v

# Reconstruir la imagen (después de cambios en el código)
docker compose build inference-service

# Ver el estado de los servicios
docker compose ps

# Ejecutar un comando en un contenedor en ejecución
docker compose exec inference-service python --version
```

### 🔧 Variables de entorno

El archivo `.env.example` contiene todas las variables con sus valores por defecto. Para usarlo:

```bash
cp .env.example .env
# Edita .env con tus valores
# Luego docker compose up -d lee .env automáticamente
```

Las más importantes para empezar:

```bash
# .env
LOG_LEVEL=INFO
WORKERS=2
API_KEYS=mi-clave-secreta-aqui
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW_SECONDS=60
REDIS_URL=redis://redis:6379
INFERENCE_BACKEND=ollama
OLLAMA_MODEL=llama3.2:1b
RAG_TOP_K=3
```

> [!WARNING]
> Nunca hagas commit de tu archivo `.env` al repositorio. Contiene secretos. El `.gitignore` ya lo excluye, pero asegúrate de que así sea.

---

## 🧪 Capítulo 8: Los tests — ¿Cómo sé que funciona?

Los tests son tu red de seguridad. Sin tests, cada cambio en el código es como cambiar un cable en un avión en vuelo con los ojos cerrados y rezar que no pase nada. Con tests, puedes hacer cambios con confianza.

### 🔬 ¿Qué se testea?

Este proyecto tiene 35+ tests unitarios en 4 archivos, diseñados para verificar **lógica pura** sin necesitar ningún servicio externo:

| Archivo | Qué testea |
|---|---|
| `test_similarity.py` | Cálculos de similitud coseno (matemáticas puras) |
| `test_schemas.py` | Validación de datos con Pydantic (contratos de la API) |
| `test_timer.py` | El context manager de latencia |
| `test_metrics_store.py` | El almacén de métricas en memoria |

### 💡 ¿Por qué no necesitan el modelo ML ni Redis?

Esta es una decisión de diseño muy importante. Los tests están escritos para verificar la **lógica** del sistema, no la infraestructura.

Por ejemplo, en `test_similarity.py`:
```python
def test_cosine_similarity_identical_vectors():
    """Dos vectores idénticos deben tener similitud 1.0"""
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == 1.0

def test_cosine_similarity_opposite_vectors():
    """Vectores opuestos deben tener similitud -1.0 (o 0.0 si se normaliza)"""
    v1 = [1.0, 0.0]
    v2 = [-1.0, 0.0]
    assert cosine_similarity(v1, v2) == pytest.approx(-1.0)
```

Esto no necesita un modelo de ML. Solo necesita NumPy y la función `cosine_similarity`. Los tests corren en milisegundos en cualquier máquina.

En `test_schemas.py`:
```python
def test_search_request_valid():
    req = SearchRequest(query="hello", top_k=5)
    assert req.query == "hello"
    assert req.top_k == 5

def test_search_request_invalid_top_k():
    """top_k debe estar entre 1 y 20"""
    with pytest.raises(ValidationError):
        SearchRequest(query="hello", top_k=25)  # Demasiado alto
```

Pydantic hace la validación; los tests verifican que los esquemas están bien definidos.

### ▶️ Cómo ejecutar los tests

```bash
# Dentro del contenedor (en producción)
docker compose exec inference-service pytest tests/ -v

# Localmente (necesitas tener las dependencias instaladas)
cd inference-service
pip install -r requirements.txt
pytest tests/ -v --tb=short
```

Ejemplo de salida:
```
tests/test_similarity.py::test_cosine_similarity_identical_vectors PASSED
tests/test_similarity.py::test_cosine_similarity_orthogonal_vectors PASSED
tests/test_schemas.py::test_embedding_request_valid PASSED
tests/test_schemas.py::test_search_request_invalid_top_k PASSED
...
35 passed in 0.42s
```

> [!TIP]
> Los tests corren en menos de 1 segundo porque no tienen que cargar modelos ML ni conectarse a bases de datos. Esto los hace perfectos para CI/CD — los puedes correr en cada commit sin demoras.

---

## 🔭 Capítulo 9: Observabilidad — Prometheus y Grafana

Desplegar un servicio sin observabilidad es como pilotar un avión sin instrumentos. Sabes que estás volando, pero no sabes a qué altitud, velocidad, ni cuánto combustible te queda.

### 🤔 ¿Qué es observabilidad?

Observabilidad es la capacidad de entender el estado interno de tu sistema mirando sus salidas. En la práctica, significa tener:

- **Métricas**: Números que cambian con el tiempo (peticiones por segundo, latencia, errores)
- **Logs**: Registros de eventos (qué pasó, cuándo, con qué datos)
- **Trazas**: El camino completo de una petición a través del sistema

Este proyecto implementa métricas (Prometheus + Grafana) y logs (JSON estructurado).

### 📊 ¿Qué métricas se recolectan?

En `services/prometheus_metrics.py` se definen contadores, histogramas y gauges:

```python
# Contadores (solo suben)
requests_total = Counter('inference_requests_total', 'Total requests', ['endpoint'])
cache_hits_total = Counter('embedding_cache_hits_total', 'Cache hits', ['level'])
rate_limit_blocks_total = Counter('rate_limit_blocks_total', 'Rate limit rejections')

# Histogramas (distribuciones de valores)
embedding_latency = Histogram('embedding_latency_seconds', 'Embedding latency',
                               buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25])
search_latency = Histogram('search_latency_seconds', 'Search latency')
rag_latency = Histogram('rag_latency_seconds', 'RAG pipeline latency')

# Gauges (suben y bajan)
active_requests = Gauge('active_requests', 'Currently processing requests')
```

### 🔄 El ciclo de métricas

```
inference-service
      │
      │ expone /metrics/prometheus (texto Prometheus)
      │
      ▼
Prometheus (cada 15 segundos scrapea la URL)
      │
      │ almacena series temporales en disco
      │
      ▼
Grafana (lee datos de Prometheus via PromQL)
      │
      │ muestra gráficas en el dashboard
      ▼
Tú (mirando el dashboard en http://localhost:3000)
```

### 📈 Accediendo a Grafana

1. Abre `http://localhost:3000` en tu navegador
2. Usuario: `admin`, Contraseña: `admin` (o los que hayas configurado en `.env`)
3. El dashboard "Inference Service" ya está pre-configurado y aparece automáticamente

Los paneles del dashboard te muestran:
- **Request Rate**: Peticiones por segundo en tiempo real
- **Latency Percentiles**: p50, p95, p99 de latencia por endpoint
- **Cache Hit Rate**: Qué porcentaje de embeddings viene de caché
- **Rate Limit Events**: Con qué frecuencia se bloquean clientes
- **Error Rate**: Porcentaje de respuestas 4xx/5xx
- **LLM Generation Time**: Cuánto tarda el LLM en responder

> [!TIP]
> El percentil p99 de latencia es tu métrica más importante para SLAs. Si el p99 es 2 segundos, significa que el 99% de tus usuarios espera 2 segundos o menos. El 1% restante puede tener mala suerte. Fíjate en ese número.

---

## 🚀 Capítulo 10: CI/CD — Automatizando la calidad

CI/CD significa *Continuous Integration / Continuous Deployment*. La idea es simple: **cada vez que subes código, un robot lo verifica automáticamente** antes de que llegue a producción.

### 🤖 ¿Qué hace GitHub Actions?

El archivo `.github/workflows/ci.yml` define un pipeline que se ejecuta automáticamente en:
- Cada push a las ramas `main` o `develop`
- Cada Pull Request apuntando a `main`

El pipeline tiene **3 jobs que se ejecutan en orden**:

```
Push a GitHub
      │
      ▼
┌─────────────┐    ✅ OK     ┌─────────────┐    ✅ OK     ┌──────────────────┐
│   1. lint   │ ──────────▶ │   2. test   │ ──────────▶ │ 3. docker-build  │
│             │             │             │             │                  │
│ ruff check  │             │ pytest -v   │             │ buildx build     │
│ ruff format │             │ 35+ tests   │             │ (no push)        │
└─────────────┘             └─────────────┘             └──────────────────┘
      │ ❌ FAIL                   │ ❌ FAIL
      ▼                           ▼
  PR bloqueada               PR bloqueada
  (no se puede mergear)       (no se puede mergear)
```

### 🔍 Job 1: Lint (calidad de código)

```yaml
- name: Lint with ruff
  run: ruff check inference-service/
  
- name: Check formatting
  run: ruff format --check inference-service/
```

**Ruff** es un linter y formateador de Python ultrarrápido (escrito en Rust). Verifica:
- Que el código sigue las convenciones de estilo (PEP 8)
- Que no hay imports sin usar
- Que no hay variables definidas pero nunca usadas
- Que el formateo es consistente (espacios, comillas, etc.)

Si ves en tu PR una ❌ en el lint, probablemente solo tienes que correr `ruff format .` en tu máquina y hacer commit de los cambios de formato.

### 🧪 Job 2: Tests

```yaml
- name: Run tests
  run: pytest inference-service/tests/ -v --tb=short
```

Corre los 35+ tests unitarios. Si alguno falla, el job falla y el PR queda bloqueado hasta que arregles el test.

### 🐳 Job 3: Docker Build

```yaml
- name: Build Docker image
  uses: docker/build-push-action@v5
  with:
    context: ./inference-service
    push: false  # Solo construye, no sube a ningún registry
    cache-from: type=gha  # Usa el caché de GitHub Actions
```

Verifica que el `Dockerfile` construye correctamente. No hace push a ningún registry — ese paso lo harías en un pipeline de CD (Continuous Deployment) separado en un proyecto real.

> [!NOTE]
> La caché de GitHub Actions (`cache-from: type=gha`) es importante. Sin ella, cada build descargaría todas las dependencias de Python desde cero (~500MB). Con la caché, las layers que no cambian se reutilizan y el build tarda segundos en lugar de minutos.

---

## 🤖 Capítulo 11: El cliente Go

Go (o Golang) es un lenguaje compilado, de tipado estático, diseñado por Google. Es conocido por producir binarios pequeños, rápidos y con muy poca latencia de arranque. En este proyecto hay un cliente CLI escrito en Go para demostrar que la API se puede consumir desde cualquier lenguaje.

### ¿Por qué Go y no Python o JavaScript?

Tres razones:

1. **Cero dependencias externas**: El cliente Go usa solo la librería estándar. Un `go build` produce un binario único que funciona en cualquier máquina Linux sin instalar nada.
2. **Contraste pedagógico**: Ver la misma API consumida desde Python y Go demuestra que la API es agnóstica al lenguaje.
3. **Realismo**: En producción, los clientes internos de microservicios frecuentemente se escriben en Go por su eficiencia.

### 📋 ¿Qué hace el cliente?

`inference-client/main.go` es un CLI con tres subcomandos:

```bash
# Verificar el estado del servicio
./inference-client health --url http://localhost:8000

# Hacer una búsqueda semántica
./inference-client search \
  --url http://localhost:8000 \
  --token tu-api-key \
  --query "¿qué es backpropagation?" \
  --top-k 3

# Ver las métricas del servicio
./inference-client metrics --url http://localhost:8000
```

### 🔨 Cómo construir y usar el cliente

```bash
# Construir el binario
cd inference-client
go build -o inference-client main.go

# O ejecutar directamente sin construir
go run main.go health --url http://localhost:8000
```

El binario resultante es ~7MB, auto-contenido, y arranca en milisegundos.

> [!NOTE]
> El cliente no tiene dependencias externas (`go.mod` solo declara el módulo). Esto es Go idiomático para herramientas CLI simples — la librería estándar de Go incluye un cliente HTTP completo, parsing de JSON, flags, y todo lo necesario.

---

## ⚡ Capítulo 12: Guía de inicio rápido (TL;DR)

Para los que quieren verlo funcionando antes de entender todo. Aquí el camino más rápido del cero al "funciona".

### 📋 Prerequisitos

- Docker y Docker Compose instalados
- 4GB de RAM libres (el modelo LLM es pesado)
- 5GB de espacio en disco (imágenes Docker + modelos)
- Puertos 8000, 3000, 6379, 9090, 11434 disponibles

### 🚀 Setup en 5 minutos

**Paso 1: Clona y configura**
```bash
git clone <url-del-repo>
cd <nombre-del-repo>
cp .env.example .env
```

**Paso 2: (Opcional) Configura una API key**
```bash
# Edita .env
echo "API_KEYS=mi-clave-de-prueba-12345" >> .env
```

**Paso 3: Arranca todos los servicios**
```bash
docker compose up -d
```

**Paso 4: Espera a que todo esté listo**

La primera vez tarda 3–10 minutos porque:
- Se descargan las imágenes Docker (~2GB en total)
- Se descarga el modelo de embeddings (~90MB)
- Se descarga el modelo LLM llama3.2:1b (~1GB)

Puedes ver el progreso con:
```bash
docker compose logs -f inference-service
```

Cuando veas `Application startup complete`, el servicio está listo.

**Paso 5: Verifica que funciona**
```bash
curl http://localhost:8000/health
```

Deberías ver:
```json
{"status": "healthy", "model_loaded": true, "chromadb_ready": true, "redis_connected": true}
```

**Paso 6: Haz tu primera búsqueda**
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer mi-clave-de-prueba-12345" \
  -d '{"query": "how do neural networks learn?", "top_k": 3}'
```

**Paso 7: Prueba RAG**
```bash
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer mi-clave-de-prueba-12345" \
  -d '{"question": "What is the difference between supervised and unsupervised learning?", "top_k": 3}'
```

### 🐛 Errores comunes y soluciones

| Error | Causa probable | Solución |
|---|---|---|
| `{"detail": "Service unavailable"}` en /search | El modelo ML aún no cargó | Espera 30s y reintenta. Verifica con /health |
| `{"detail": "Unauthorized"}` | API key incorrecta o no enviada | Verifica el header `Authorization: Bearer <tu-key>` |
| `{"detail": "Rate limit exceeded"}` | Demasiadas peticiones | Espera 60s o aumenta `RATE_LIMIT_REQUESTS` en `.env` |
| Puerto 8000 ocupado | Otro proceso usa ese puerto | Cambia el mapeo en `docker-compose.yml` a `"8001:8000"` |
| El contenedor de Ollama reinicia | Falta RAM | Asegúrate de tener 4GB libres. Prueba con un modelo más pequeño |
| Grafana no muestra datos | Prometheus aún no scrapeó | Espera 30s para el primer scrape |

---

## 🗺️ Capítulo 13: Flujo de datos completo — De la petición a la respuesta

Este capítulo traza el viaje de una petición `/rag` de principio a fin. Es como seguir una pizza desde que el cliente la pide hasta que llega a su puerta.

### 🍕 La petición

```bash
POST /rag
Authorization: Bearer mi-clave
Content-Type: application/json

{"question": "¿Cómo funciona el algoritmo de k-means?", "top_k": 3}
```

### 📍 Paso a paso

```
1. 🌐 ENTRADA HTTP
   ─────────────────────────────────────────────────────
   La petición llega a Uvicorn (ASGI server) en el puerto 8000.
   Uvicorn la encola y la pasa a uno de los 2 workers.

2. 🔐 MIDDLEWARE: AUTENTICACIÓN (auth.py)
   ─────────────────────────────────────────────────────
   Lee el header "Authorization: Bearer mi-clave"
   Busca "mi-clave" en la lista API_KEYS.
   ✅ Encontrado → continúa.
   ❌ No encontrado → devuelve 401 Unauthorized aquí mismo.

3. 🚦 MIDDLEWARE: RATE LIMITING (rate_limiter.py)
   ─────────────────────────────────────────────────────
   Obtiene la IP del cliente (ej: "172.18.0.1")
   Consulta Redis: "ratelimit:172.18.0.1"
   ZREMRANGEBYSCORE: elimina peticiones de hace más de 60s
   ZCARD: cuenta peticiones en la ventana → 23
   ZADD: añade timestamp actual
   23 < 60 → ✅ continúa.

4. 🛣️ ROUTING
   ─────────────────────────────────────────────────────
   FastAPI identifica el endpoint "/rag" (POST)
   Valida el body con Pydantic:
     ✅ "question" es str no vacío
     ✅ "top_k" es int entre 1 y 20 (default 3)
   Llama a la función handler.

5. 🔍 RAG STEP 1: RETRIEVE (search_service.py)
   ─────────────────────────────────────────────────────
   
   5a. EMBEDDING DE LA PREGUNTA (embedding_service.py)
       hash("¿Cómo funciona el algoritmo de k-means?") → "abc123"
       
       L1 check: ¿"abc123" en dict local? → MISS
       L2 check: ¿"emb:abc123" en Redis? → MISS (primera vez)
       L3: Llama a SentenceTransformer.encode(texto)
          → [0.12, -0.34, ..., 0.56] (384 floats) en ~50ms
       Guarda en Redis con TTL 86400
       Guarda en dict L1
   
   5b. BÚSQUEDA EN CHROMADB (vector_store.py)
       query_embeddings=[[0.12, -0.34, ..., 0.56]]
       n_results=3
       
       ChromaDB hace búsqueda HNSW sobre los 15 documentos:
       Resultado:
         doc_005: score=0.891 "K-means es un algoritmo de clustering..."
         doc_008: score=0.834 "El clustering agrupa puntos similares..."
         doc_012: score=0.762 "Los centroides son los representantes..."

6. 📝 RAG STEP 2: AUGMENT (rag_service.py)
   ─────────────────────────────────────────────────────
   Construye el prompt:
   
   "Basándote únicamente en el siguiente contexto, responde la pregunta.
   
   Contexto:
   [1] (relevancia: 0.89): K-means es un algoritmo de clustering no supervisado
       que agrupa n puntos en k clusters minimizando la varianza intra-cluster...
   [2] (relevancia: 0.83): El clustering es la tarea de agrupar objetos similares...
   [3] (relevancia: 0.76): Los centroides son los puntos que representan el centro...
   
   Pregunta: ¿Cómo funciona el algoritmo de k-means?
   
   Respuesta:"

7. 🤖 RAG STEP 3: GENERATE (ollama_client.py)
   ─────────────────────────────────────────────────────
   HTTP POST a http://ollama:11434/api/generate
   {
     "model": "llama3.2:1b",
     "prompt": "<el prompt anterior>",
     "temperature": 0.3,
     "max_tokens": 512,
     "stream": false
   }
   
   Ollama procesa el prompt con llama3.2:1b (~1-3 segundos)
   
   Respuesta: "K-means es un algoritmo iterativo que funciona así:
   1. Inicializa k centroides aleatoriamente...
   2. Asigna cada punto al centroide más cercano...
   3. Recalcula los centroides como la media..."

8. 📊 MÉTRICAS (prometheus_metrics.py)
   ─────────────────────────────────────────────────────
   Incrementa counter: inference_requests_total{endpoint="/rag"}
   Registra histograma: rag_latency_seconds con valor total
   Registra histograma: embedding_latency_seconds
   Registra histograma: search_latency_seconds
   Registra: embedding_cache_misses (L1 y L2 fallaron)

9. 📋 LOGGING (logger.py)
   ─────────────────────────────────────────────────────
   Emite log JSON estructurado:
   {
     "timestamp": "2026-02-27T10:23:45Z",
     "level": "INFO",
     "endpoint": "/rag",
     "latency_ms": 1342.7,
     "top_k": 3,
     "cache_hit": false,
     "client_ip": "172.18.0.1"
   }

10. 📤 RESPUESTA HTTP
    ─────────────────────────────────────────────────────
    FastAPI serializa el objeto RAGResponse a JSON:
    {
      "answer": "K-means es un algoritmo iterativo...",
      "sources": [
        {"id": "doc_005", "score": 0.891, "text": "K-means es..."},
        {"id": "doc_008", "score": 0.834, "text": "El clustering..."},
        {"id": "doc_012", "score": 0.762, "text": "Los centroides..."}
      ],
      "retrieval_latency_ms": 51.2,
      "generation_latency_ms": 1287.3,
      "total_latency_ms": 1342.7
    }
    
    HTTP 200 OK → de vuelta al cliente.
```

### ⏱️ Desglose de tiempos

| Paso | Tiempo típico |
|---|---|
| Auth + Rate Limit | < 1ms |
| Embedding (L3, primera vez) | ~50ms |
| Embedding (L2, caché Redis) | ~0.5ms |
| Búsqueda HNSW ChromaDB | ~1ms |
| Generación LLM (llama3.2:1b) | ~1000–3000ms |
| Serialización JSON + respuesta | < 1ms |
| **TOTAL (primera vez)** | **~1100–3100ms** |
| **TOTAL (con caché de embedding)** | **~1050–3050ms** |

La conclusión obvia: el LLM domina el tiempo de respuesta. Las optimizaciones de caché en embedding son importantes para el throughput, pero el cuello de botella real en RAG siempre es la generación.

---

## 📚 Capítulo 14: Glosario — Las palabras raras explicadas

Porque el mundo de ML está lleno de palabras que suenan intimidantes y luego resultan ser conceptos simples con nombres complicados.

| Término | Definición humana |
|---|---|
| **ASGI** | Protocolo que permite a Python manejar peticiones async. Como WSGI pero para async. Uvicorn lo implementa. |
| **Bearer Token** | Un token que "porta" (lleva) tu identidad. Lo mandas en la cabecera HTTP y el servidor verifica que es válido. Como un pase de backstage. |
| **ChromaDB** | Base de datos vectorial. Guarda vectores y los busca eficientemente por similitud. |
| **Corpus** | Colección de documentos. El "corpus semilla" son los 15 documentos pre-cargados. |
| **Cosine Similarity** | Medida de similitud entre vectores. 1 = idénticos, 0 = sin relación, -1 = opuestos. Mide el ángulo entre los vectores, no la distancia. |
| **Docker Compose** | Herramienta para definir y correr aplicaciones multi-contenedor. Un archivo YAML que describe todos los servicios. |
| **Embedding** | Representación numérica de texto en forma de vector. Captura el significado semántico. |
| **FastAPI** | Framework web de Python, moderno y async. Genera documentación Swagger automáticamente. |
| **Grafana** | Herramienta de visualización de métricas. Los dashboards bonitos con gráficas. |
| **HNSW** | Hierarchical Navigable Small World. Algoritmo de búsqueda aproximada de vecinos más cercanos. Muy eficiente para millones de vectores. |
| **Inference** | En ML, "inference" = usar un modelo ya entrenado para hacer predicciones. Lo opuesto a "training". |
| **L1/L2/L3 Cache** | Niveles de caché. L1 más rápido pero pequeño (memoria local), L2 compartida (Redis), L3 la fuente original (modelo ML). |
| **Latency** | Tiempo que tarda una operación. Se mide en ms (milisegundos). |
| **LLM** | Large Language Model. Modelo de lenguaje grande. ChatGPT, Llama, etc. Genera texto. |
| **Middleware** | Código que se ejecuta en cada petición HTTP, entre la entrada y el handler. Como un filtro. |
| **Ollama** | Servidor local para correr LLMs. Como tener ChatGPT en tu máquina. |
| **Pydantic** | Librería de Python para validar datos. Defines un esquema y valida que los datos cumplen el contrato. |
| **Prometheus** | Sistema de monitorización que scrapea métricas y las almacena como series temporales. |
| **RAG** | Retrieval-Augmented Generation. Busca documentos relevantes y los usa como contexto para el LLM. |
| **Rate Limiting** | Limitar el número de peticiones por cliente en un período de tiempo. Evita abuso. |
| **Redis** | Base de datos en memoria, ultrarrápida. Aquí se usa para caché y rate limiting. |
| **SentenceTransformer** | Modelo de ML que convierte frases en embeddings. El que usa este proyecto es `all-MiniLM-L6-v2`. |
| **Semantic Search** | Búsqueda por significado, no por palabras exactas. "machine learning" y "aprendizaje automático" son iguales semánticamente. |
| **Sliding Window** | Técnica de rate limiting donde la ventana de tiempo se mueve con cada petición, en lugar de reiniciarse en intervalos fijos. |
| **Throughput** | Cuántas peticiones por segundo puede manejar el sistema. |
| **Upsert** | Insert + Update. Si el registro existe, actualiza; si no existe, crea. |
| **Uvicorn** | Servidor ASGI para Python. Ejecuta FastAPI. Puede correr con múltiples workers. |
| **Vector** | Lista de números. En este contexto, la representación numérica de un texto. |
| **Vector Store** | Base de datos optimizada para almacenar y buscar vectores. ChromaDB es un vector store. |
| **vLLM** | Motor de inferencia de LLMs optimizado para producción (alto throughput). Alternativa a Ollama para producción. |
| **Worker** | Proceso separado que maneja peticiones. Con WORKERS=2, hay 2 procesos Python manejando peticiones en paralelo. |
| **ZSET** | Sorted Set de Redis. Conjunto de elementos ordenados por un score numérico. Ideal para el sliding window de rate limiting. |

---

## 🎓 Capítulo 15: ¿Qué aprendiste? Resumen final

Has llegado al final. Felicidades, de verdad. Si has leído hasta aquí, eres 100x más capaz de trabajar con sistemas de inferencia ML que cuando empezaste.

### 🏆 Lo que has aprendido

**Conceptos fundamentales de ML en producción:**
- Los embeddings son representaciones numéricas del significado del texto
- La búsqueda semántica encuentra documentos por similitud de significado, no por palabras exactas
- RAG hace que los LLMs sean más precisos al darles contexto relevante antes de generar
- Los sistemas de producción necesitan caché, rate limiting, autenticación y observabilidad

**Patrones de arquitectura:**
- El **patrón Factory** permite cambiar implementaciones (Ollama/vLLM) con un cambio de config
- El **sistema de caché multinivel** balancea velocidad, costo y compartición de datos
- La arquitectura de **microservicios** permite escalar y mantener cada componente independientemente
- El **lifespan pattern** de FastAPI gestiona el ciclo de vida del servicio de forma elegante

**Habilidades prácticas:**
- Cómo leer y entender un proyecto backend multi-servicio
- Cómo usar Docker Compose para orquestar servicios
- Cómo interpretar métricas de Prometheus en Grafana
- Cómo diseñar tests que no dependan de infraestructura externa
- Cómo implementar rate limiting con Redis ZSETs
- Cómo construir un pipeline RAG desde cero

### 🗺️ ¿A dónde ir desde aquí?

Este proyecto es un excelente punto de partida. Aquí hay caminos para seguir aprendiendo:

**Si quieres profundizar en ML/AI:**
- Experimenta con diferentes modelos de embeddings (prueba `all-mpnet-base-v2` para mayor calidad)
- Prueba con diferentes LLMs en Ollama (`mistral`, `codellama`, etc.)
- Aprende sobre fine-tuning de modelos de embeddings para tu dominio específico

**Si quieres mejorar el sistema:**
- Añade autenticación por usuario (no solo por API key global)
- Implementa un sistema de feedback para mejorar la relevancia de búsqueda
- Añade soporte para documentos PDF/Word (parsing + chunking)
- Implementa `reranking` para mejorar la calidad de los resultados de búsqueda

**Si quieres aprender más sobre infraestructura:**
- Estudia cómo desplegar esto en Kubernetes
- Aprende sobre auto-scaling basado en métricas de Prometheus
- Explora cómo configurar vLLM para producción de alto throughput
- Añade distributed tracing con OpenTelemetry

**Si quieres mejorar la calidad del código:**
- Añade tests de integración (sí, los que necesitan Redis y ChromaDB)
- Implementa property-based testing con Hypothesis
- Añade type stubs y verifica con mypy

### 💡 El mensaje final

Este proyecto demuestra algo importante: **ML en producción no es magia**. Es ingeniería de software. Las mismas habilidades que usas para construir una API REST — separación de responsabilidades, caché, autenticación, tests, CI/CD — se aplican exactamente igual cuando el servicio hace inferencia ML.

La única diferencia es que hay nuevas piezas (el modelo de embeddings, el vector store, el LLM) que necesitas entender. Y ahora las entiendes.

El resto es solo código.

---

> *"La mejor forma de aprender es haciendo. Clona el repo, rómpelo, arréglalo, y añade algo nuevo. Eso es lo que hace un buen ingeniero."*

---

**Fin de la guía** | Versión 1.0 | Escrita con cariño para devs curiosos

