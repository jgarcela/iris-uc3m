"""
Precios de referencia por modelo (USD por 1M de tokens) para estimar el coste
por artículo del Experimento 21. Editar según tarifas vigentes.

`cache_read` se factura como input con descuento; si no se conoce, se asimila a input.
Modelos locales (Ollama): coste 0 (self-hosted).
"""
from __future__ import annotations

# (input, output, cache_read) USD por 1M tokens. Verificado 2026-07-20.
# Fuentes: docs oficiales Anthropic y Google; OpenAI ver nota de modelos legacy.
PRECIOS = {
    # OpenAI — OJO: retirados de la página de precios principal en 2026 (legacy).
    # Los snapshots antiguos de GPT-5 se retiran de la API el 2026-12-11.
    "gpt-5.4-nano":           (0.20, 1.25, 0.02),    # vigente en el catálogo actual de OpenAI
    "gpt-4o-mini":            (0.15, 0.60, 0.075),   # input verificado; output/cache tarifa histórica
    "gpt-5-nano":             (0.05, 0.40, 0.005),   # input verificado; output/cache sin confirmar
    # Anthropic — verificado en docs oficiales (Haiku 4.5: $1 / $5; cache read ≈0.1x).
    "claude-haiku-4-5-20251001": (1.00, 5.00, 0.10),
    "claude-haiku-4-5":       (1.00, 5.00, 0.10),
    # Google — verificado en ai.google.dev/gemini-api/docs/pricing (tier estándar).
    "gemini-3.1-flash-lite":  (0.25, 1.50, 0.025),
    "gemini-2.5-flash":       (0.30, 2.50, 0.075),
}
LOCAL_HINTS = ("gemma", "qwen", "llama", "mistral", "deepseek", "phi")


def _tarifa(modelo: str) -> tuple[float, float, float] | None:
    if modelo in PRECIOS:
        return PRECIOS[modelo]
    if any(h in modelo.lower() for h in LOCAL_HINTS):
        return (0.0, 0.0, 0.0)  # local self-hosted
    return None  # desconocido → coste no estimable


def calcular_coste(modelo: str, prompt_tokens: int, completion_tokens: int,
                   cache_read_tokens: int = 0, cache_creation_tokens: int = 0) -> float | None:
    """Coste en USD de una acumulación de tokens. None si el modelo no tiene tarifa."""
    tarifa = _tarifa(modelo)
    if tarifa is None:
        return None
    p_in, p_out, p_cache = tarifa
    input_normal = max(prompt_tokens - cache_read_tokens, 0)
    coste = (
        input_normal * p_in
        + cache_read_tokens * p_cache
        + cache_creation_tokens * p_in
        + completion_tokens * p_out
    ) / 1_000_000
    return round(coste, 6)
