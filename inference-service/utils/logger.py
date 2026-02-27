"""
logger.py
---------
Configuración centralizada de logging estructurado en formato JSON.

Un solo logger para toda la aplicación. Formato JSON permite
que los logs sean consumidos por sistemas como Datadog, CloudWatch o Loki
con filtrado por campos específicos (ej: duration_ms > 100).

Uso:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("event_name", extra={"duration_ms": 45.2, "cache_hit": True})
"""

import logging
import json
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Formateador que serializa cada log como una línea JSON.

    Campos fijos en cada línea:
      - timestamp: ISO 8601
      - level: DEBUG / INFO / WARNING / ERROR
      - logger: nombre del módulo que generó el log
      - event: mensaje principal (el string que se pasa a logger.info())
      - (cualquier campo extra pasado via extra={...})
    """

    def format(self, record: logging.LogRecord) -> str:
        # Campos base
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        # Agregar campos extra si fueron pasados con extra={...}
        # Excluye los atributos estándar de LogRecord para no duplicarlos
        _standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "taskName",
        }
        for key, value in record.__dict__.items():
            if key not in _standard_attrs:
                log_entry[key] = value

        # Si hay excepción, incluirla
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """
    Retorna un logger configurado con formato JSON.

    Args:
        name: Nombre del módulo (usar __name__).

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)

    # Evitar agregar múltiples handlers si get_logger se llama varias veces
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        # Evitar que los logs suban al root logger (evita duplicados)
        logger.propagate = False

    return logger
