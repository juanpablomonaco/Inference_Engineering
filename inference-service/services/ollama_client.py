"""
ollama_client.py
----------------
Cliente HTTP para el servicio Ollama (LLM local).

Responsabilidad única: comunicación con la API de Ollama.
No sabe nada de RAG, embeddings ni búsqueda semántica.

Ollama expone una API compatible con OpenAI en:
  POST /api/chat    → chat con historial de mensajes
  POST /api/generate → generación simple
  GET  /api/tags    → modelos disponibles
  GET  /api/health  → estado del servicio

Por qué no usar el SDK oficial de OpenAI:
  Para este proyecto educativo, implementar el cliente HTTP
  directamente con httpx enseña exactamente qué ocurre en la red.
  La evolución a openai SDK es trivial: solo cambia este archivo.

Timeouts:
  La generación de texto puede tardar 10-60 segundos en CPU.
  Usamos timeout generoso de 120s para modelos livianos.
"""

import httpx
from utils.logger import get_logger

logger = get_logger(__name__)

GENERATE_TIMEOUT = 120.0  # segundos — generación en CPU puede ser lenta


class OllamaUnavailableError(Exception):
    """Raised cuando Ollama no está disponible o el modelo no está cargado."""

    pass


class OllamaClient:
    """
    Cliente HTTP para Ollama.

    Attributes:
        base_url: URL base del servicio (ej: http://ollama:11434)
        model: Nombre del modelo a usar (ej: llama3.2:1b)
    """

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.Client(timeout=GENERATE_TIMEOUT)

    def health_check(self) -> bool:
        """
        Verifica que Ollama esté disponible y el modelo esté cargado.

        Returns:
            True si el servicio responde y el modelo está disponible.
        """
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            # Verificar que el modelo esté disponible (con o sin tag :latest)
            model_available = any(
                self.model in m or m.startswith(self.model.split(":")[0])
                for m in models
            )
            logger.info(
                "ollama_health_check",
                extra={
                    "available_models": models,
                    "target_model_ready": model_available,
                },
            )
            return model_available
        except Exception as e:
            logger.warning("ollama_health_check_failed", extra={"error": str(e)})
            return False

    def pull_model(self) -> None:
        """
        Descarga el modelo si no está disponible localmente.
        Llamado al startup si health_check() retorna False.
        Puede tardar varios minutos dependiendo del tamaño del modelo.
        """
        logger.info("ollama_model_pull_started", extra={"model": self.model})
        try:
            # stream=True para no hacer timeout en el pull
            with self._client.stream(
                "POST",
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                timeout=600.0,  # 10 minutos para descarga
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if '"status"' in line:
                        logger.info("ollama_pull_progress", extra={"line": line[:100]})
            logger.info("ollama_model_pull_completed", extra={"model": self.model})
        except Exception as e:
            logger.error(
                "ollama_model_pull_failed", extra={"model": self.model, "error": str(e)}
            )
            raise OllamaUnavailableError(f"Failed to pull model {self.model}: {e}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Genera una respuesta usando el chat API de Ollama.

        Usa el formato de mensajes compatible con OpenAI:
          [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

        Args:
            system_prompt: Instrucciones del sistema (rol del asistente).
            user_prompt: Mensaje del usuario (query + contexto RAG).

        Returns:
            Texto de la respuesta generada.

        Raises:
            OllamaUnavailableError: Si Ollama no responde o hay error de generación.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,  # Respuesta completa, no streaming
            "options": {
                "temperature": 0.3,  # Baja temperatura para respuestas más factuales
                "num_predict": 512,  # Máximo de tokens en la respuesta
            },
        }

        try:
            response = self._client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            answer = data["message"]["content"].strip()
            return answer

        except httpx.TimeoutException:
            raise OllamaUnavailableError(
                f"Ollama generation timed out after {GENERATE_TIMEOUT}s. "
                "The model may be loading or the CPU is slow."
            )
        except httpx.HTTPStatusError as e:
            raise OllamaUnavailableError(
                f"Ollama returned HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise OllamaUnavailableError(f"Ollama generation failed: {e}")

    def __del__(self) -> None:
        """Cierra el cliente HTTP al destruir la instancia."""
        try:
            self._client.close()
        except Exception:
            pass
