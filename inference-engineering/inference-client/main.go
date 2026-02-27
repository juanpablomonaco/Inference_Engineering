// main.go
// --------
// Cliente Go para el servicio de inferencia.
//
// Propósito educativo:
//   Este cliente te enseña cómo los microservicios se comunican entre sí
//   a través de contratos HTTP explícitos. Go es fuertemente tipado, por
//   lo que debes modelar exactamente la estructura de la respuesta —
//   esto hace visible el contrato entre servicios.
//
// Funcionalidades:
//   - Verifica el health del servicio antes de usarlo
//   - Envía una query al endpoint /search
//   - Muestra el resultado y métricas del servicio
//   - Manejo de errores HTTP con contexto claro
//
// Uso:
//   go run main.go
//   go run main.go -query "what is a transformer model?"
//   go run main.go -url http://localhost:8000 -query "explain embeddings"

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// ---------------------------------------------------------------------------
// Structs — contratos de datos (espejo de los schemas Python)
// ---------------------------------------------------------------------------

// SearchRequest es el cuerpo del POST /search
type SearchRequest struct {
	Query string `json:"query"`
}

// SearchResponse es la respuesta del POST /search
type SearchResponse struct {
	Result    string  `json:"result"`
	Score     float64 `json:"score"`
	ElapsedMs float64 `json:"elapsed_ms"`
}

// HealthResponse es la respuesta del GET /health
type HealthResponse struct {
	Status             string `json:"status"`
	ModelLoaded        bool   `json:"model_loaded"`
	CorpusInitialized  bool   `json:"corpus_initialized"`
	CacheReady         bool   `json:"cache_ready"`
}

// MetricsResponse es la respuesta del GET /metrics
type MetricsResponse struct {
	TotalRequests        int     `json:"total_requests"`
	TotalEmbeddingCalls  int     `json:"total_embedding_calls"`
	TotalSearchCalls     int     `json:"total_search_calls"`
	AvgEmbeddingMs       float64 `json:"avg_embedding_ms"`
	AvgSearchMs          float64 `json:"avg_search_ms"`
	CacheHits            int     `json:"cache_hits"`
	CacheMisses          int     `json:"cache_misses"`
	CacheHitRatio        float64 `json:"cache_hit_ratio"`
}

// ---------------------------------------------------------------------------
// HTTP Client helper
// ---------------------------------------------------------------------------

// InferenceClient encapsula la comunicación con el servicio de inferencia.
// Mantiene la URL base y el cliente HTTP con timeout configurado.
type InferenceClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewInferenceClient crea un cliente con timeout de 30 segundos.
// El timeout es importante: sin él, una request colgada bloquea para siempre.
func NewInferenceClient(baseURL string) *InferenceClient {
	return &InferenceClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// checkHealth verifica si el servicio está listo para recibir requests.
// Retorna error si el servicio no está healthy o no responde.
func (c *InferenceClient) checkHealth() (*HealthResponse, error) {
	resp, err := c.httpClient.Get(c.baseURL + "/health")
	if err != nil {
		return nil, fmt.Errorf("failed to reach service at %s: %w", c.baseURL, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read health response body: %w", err)
	}

	var health HealthResponse
	if err := json.Unmarshal(body, &health); err != nil {
		return nil, fmt.Errorf("failed to parse health response: %w", err)
	}

	// HTTP 503 significa que el servicio existe pero no está listo
	if resp.StatusCode != http.StatusOK {
		return &health, fmt.Errorf(
			"service is not ready (HTTP %d): model_loaded=%v, corpus_initialized=%v",
			resp.StatusCode, health.ModelLoaded, health.CorpusInitialized,
		)
	}

	return &health, nil
}

// search envía una query al endpoint /search y retorna el resultado.
func (c *InferenceClient) search(query string) (*SearchResponse, error) {
	// Serializar el request body
	reqBody := SearchRequest{Query: query}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to serialize request: %w", err)
	}

	// Construir y enviar la request HTTP
	resp, err := c.httpClient.Post(
		c.baseURL+"/search",
		"application/json",
		bytes.NewBuffer(payload),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to send search request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read search response body: %w", err)
	}

	// Manejo de errores HTTP (4xx, 5xx)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf(
			"search request failed (HTTP %d): %s",
			resp.StatusCode, string(body),
		)
	}

	// Deserializar la respuesta
	var searchResp SearchResponse
	if err := json.Unmarshal(body, &searchResp); err != nil {
		return nil, fmt.Errorf("failed to parse search response: %w", err)
	}

	return &searchResp, nil
}

// getMetrics recupera las métricas actuales del servicio.
func (c *InferenceClient) getMetrics() (*MetricsResponse, error) {
	resp, err := c.httpClient.Get(c.baseURL + "/metrics")
	if err != nil {
		return nil, fmt.Errorf("failed to reach metrics endpoint: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read metrics response body: %w", err)
	}

	var metricsResp MetricsResponse
	if err := json.Unmarshal(body, &metricsResp); err != nil {
		return nil, fmt.Errorf("failed to parse metrics response: %w", err)
	}

	return &metricsResp, nil
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	// Flags de configuración — permiten cambiar URL y query sin recompilar
	baseURL := flag.String("url", "http://localhost:8000", "URL base del servicio de inferencia")
	query := flag.String("query", "What is a neural network?", "Query para búsqueda semántica")
	flag.Parse()

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  Inference Engineering — Go Client")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("  Service URL : %s\n", *baseURL)
	fmt.Printf("  Query       : %s\n", *query)
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	client := NewInferenceClient(*baseURL)

	// ── [1] Health check ──────────────────────────────────────────────────
	fmt.Println("\n[1] Checking service health...")
	health, err := client.checkHealth()
	if err != nil {
		fmt.Fprintf(os.Stderr, "    ERROR: %v\n", err)
		fmt.Fprintln(os.Stderr, "\n    Make sure the service is running:")
		fmt.Fprintln(os.Stderr, "    $ docker-compose up")
		os.Exit(1)
	}
	fmt.Printf("    Status             : %s\n", health.Status)
	fmt.Printf("    Model loaded       : %v\n", health.ModelLoaded)
	fmt.Printf("    Corpus initialized : %v\n", health.CorpusInitialized)
	fmt.Printf("    Cache ready        : %v\n", health.CacheReady)

	// ── [2] Semantic search ───────────────────────────────────────────────
	fmt.Printf("\n[2] Sending search query...\n")
	fmt.Printf("    Query: \"%s\"\n\n", *query)

	start := time.Now()
	result, err := client.search(*query)
	clientElapsed := time.Since(start).Milliseconds()

	if err != nil {
		fmt.Fprintf(os.Stderr, "    ERROR: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("    ┌─ Result ──────────────────────────────────────")
	fmt.Printf("    │  %s\n", result.Result)
	fmt.Println("    └───────────────────────────────────────────────")
	fmt.Printf("\n    Similarity score   : %.4f\n", result.Score)
	fmt.Printf("    Service latency    : %.2f ms\n", result.ElapsedMs)
	fmt.Printf("    Client RTT         : %d ms\n", clientElapsed)

	// ── [3] Métricas del servicio ─────────────────────────────────────────
	fmt.Println("\n[3] Service metrics snapshot...")
	metrics, err := client.getMetrics()
	if err != nil {
		// Las métricas no son críticas — solo advertimos
		fmt.Printf("    WARNING: Could not fetch metrics: %v\n", err)
	} else {
		fmt.Printf("    Total requests     : %d\n", metrics.TotalRequests)
		fmt.Printf("    Total searches     : %d\n", metrics.TotalSearchCalls)
		fmt.Printf("    Avg embedding ms   : %.2f ms\n", metrics.AvgEmbeddingMs)
		fmt.Printf("    Avg search ms      : %.2f ms\n", metrics.AvgSearchMs)
		fmt.Printf("    Cache hits         : %d\n", metrics.CacheHits)
		fmt.Printf("    Cache misses       : %d\n", metrics.CacheMisses)
		fmt.Printf("    Cache hit ratio    : %.1f%%\n", metrics.CacheHitRatio*100)
	}

	fmt.Println("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("  Done.")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
}
