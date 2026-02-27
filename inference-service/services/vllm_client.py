"""
vllm_client.py
--------------
Cliente HTTP para vLLM — backend de producción para generación LLM.

Por qué vLLM sobre Ollama en producción:
  - PagedAttention: gestión eficiente de memoria KV-cache
  - Continuous batching: agrupa múltiples requests en un solo forward pass
  - Throughput: 10-30x más alto que Ollama con múltiples usuarios concurrentes
  - API OpenAI-compatible: drop-in replacement, misma interfaz que Ollama

Cuándo usar cada uno:
  OLLAMA → desarrollo local, una sola persona, CPU, modelos pequeños (~1-7B)
  vLLM   → staging/producción, múltiples usuarios, GPU, modelos 7B+

Configuración:
  INFERENCE_BACKEND=vllm        → activa este cliente en rag_service.py
  VLLM_BASE_URL=http://vllm:8001  → URL del servidor vLLM
  OLLAMA_MODEL=meta-llama/Llama-3.1-8B-Instruct  → modelo a cargar en vLLM

Cómo correr vLLM localmente (requiere GPU con 8GB+ VRAM):
  docker run --gpus all -p 8001:8000 vllm/vllm-openai:latest \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --max-model-len 4096

La misma interfaz que OllamaClient (health_check, generate) permite
swapear entre backends sin cambiar rag_service.py.
"""

import os
import httpx
from utils.logger import get_logger

logger = get_logger(__name__)

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8001")
GENERATE_TIMEOUT = 120.0


class VLLMUnavailableError(Exception):
    """Raised cuando vLLM no está disponible."""
    pass


class VLLMClient:
    """
    Cliente para vLLM usando su API OpenAI-compatible (/v1/chat/completions).

    Misma interfaz pública que OllamaClient:
      - health_check() → bool
      - generate(system_prompt, user_prompt) → str

    Esto permite swapear entre Ollama y vLLM en rag_service.py
    simplemente cambiando la instancia del cliente.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=GENERATE_TIMEOUT)

    def health_check(self) -> bool:
        """
        Verifica que vLLM esté disponible consultando /v1/models.
        Retorna True si el modelo configurado está cargado.
        """
        try:
            resp = self._client.get(f"{self.base_url}/v1/models", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            model_ids = [m["id"] for m in data.get("data", [])]
            model_ready = any(self.model in mid for mid in model_ids)
            logger.info(
                "vllm_health_check",
                extra={"available_models": model_ids, "target_model_ready": model_ready},
            )
            return model_ready
        except Exception as e:
            logger.warning("vllm_health_check_failed", extra={"error": str(e)})
            return False

    def pull_model(self) -> None:
        """
        vLLM carga el modelo al arrancar el servidor, no hay pull dinámico.
        Este método existe solo para mantener la misma interfaz que OllamaClient.
        """
        logger.info(
            "vllm_pull_model_noop",
            extra={"note": "vLLM loads models at server startup, not dynamically."},
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Genera una respuesta usando la API OpenAI-compatible de vLLM.

        Endpoint: POST /v1/chat/completions
        Formato: idéntico a OpenAI Chat API.

        Args:
            system_prompt: Instrucciones del sistema.
            user_prompt: Mensaje del usuario con contexto RAG.

        Returns:
            Texto de la respuesta generada.

        Raises:
            VLLMUnavailableError: Si vLLM no responde o hay error.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 512,
            "temperature": 0.3,
            "stream": False,
        }

        try:
            response = self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["choices"][0]["message"]["content"].strip()
            return answer

        except httpx.TimeoutException:
            raise VLLMUnavailableError(
                f"vLLM generation timed out after {GENERATE_TIMEOUT}s."
            )
        except httpx.HTTPStatusError as e:
            raise VLLMUnavailableError(
                f"vLLM returned HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise VLLMUnavailableError(f"vLLM generation failed: {e}")

    def __del__(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
