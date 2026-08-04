"""
Experimento 22 — Confianza / P(Sí).

Clasificador B0 (metodología inyectada, 1 llamada por variable) que además pide
la probabilidad prob_si (0-100) de que la variable esté presente. Reutiliza —sin
modificarlos— la metodología (skills), el enrutado multi-proveedor (utils) y la
tabla de costes del Experimento 21; solo cambia el prompt (añade prob_si) y el
parseo. No depende del bucle de agentes ni de herramientas.

El objeto de estudio no es la arquitectura, sino la SEÑAL de confianza del modelo:
calibración, barrido de umbral (dev→test) y confianza en los desacuerdos con el GT.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EXP22_DIR = Path(__file__).resolve().parent
EXPERIMENTOS_DIR = EXP22_DIR.parent.parent
EXP21_DIR = EXPERIMENTOS_DIR / "experiments" / "experimento_21_agentskills"
for p in (str(EXPERIMENTOS_DIR), str(EXP21_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import consultar_ollama, get_consumo_llamada  # noqa: E402
import tools   # noqa: E402  (de exp21: read_skill, SKILLS_VARIABLE)
import costes  # noqa: E402  (de exp21: calcular_coste)

# Mismo texto de metodología que el B0 del exp21, más la petición de prob_si.
SYSTEM = """Eres una experta en análisis de género en medios y clasificas ÚNICAMENTE la variable «{variable}».

A continuación tienes la metodología completa que debes aplicar.

--- METODOLOGÍA: {variable} ---
{metodologia}
--- FIN METODOLOGÍA ---

Aplica la metodología y responde con UNA sola línea, exactamente en esta forma:

  FINAL: {{"codigo": <n>, "prob_si": <0-100>, "explicacion": "...", "evidencias": [literales o []]}}

`prob_si` es tu probabilidad (0 a 100) de que la variable ESTÉ PRESENTE (código > 1),
con independencia del `codigo` que asignes: 0 = con seguridad NO; 100 = con seguridad
SÍ; 50 = caso fronterizo. Sé sincera y usa valores intermedios cuando dudes.
No añadas nada más ni mezcles otras variables."""


def _parse_final(salida: str) -> dict | None:
    m = re.search(r"FINAL:\s*(\{.*\})", salida, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def clasificar(variable: str, texto: str, modelo: str,
               temperature: float = 0.1) -> tuple[dict, dict]:
    """Clasifica una variable pidiendo codigo + prob_si. Devuelve (resultado, traza)."""
    system = SYSTEM.format(variable=variable, metodologia=tools.read_skill(variable))
    prompt = (system + "\n\n=== TEXTO A CLASIFICAR ===\n" + texto
              + "\n=== FIN TEXTO ===\n\nTu respuesta:")
    salida = consultar_ollama(prompt, modelo=modelo, temperature=temperature)
    c = get_consumo_llamada()
    traza = {
        "prompt_tokens": c.get("prompt_tokens", 0),
        "completion_tokens": c.get("completion_tokens", 0),
        "cache_read_tokens": c.get("cache_read_tokens", 0),
        "error": None,
    }
    traza["coste_usd"] = costes.calcular_coste(
        modelo, traza["prompt_tokens"], traza["completion_tokens"],
        traza["cache_read_tokens"], 0)

    data = _parse_final(salida)
    if not isinstance(data, dict):
        traza["error"] = "sin_final"
        return {"codigo": 1, "prob_si": None, "explicacion": salida[:200], "evidencias": []}, traza

    codigo = data.get("codigo", 1)
    prob = data.get("prob_si")
    try:
        prob = max(0.0, min(100.0, float(prob)))
    except (TypeError, ValueError):
        prob = None
    evid = data.get("evidencias") or []
    if not isinstance(evid, list):
        evid = [str(evid)]
    evid = [e for e in evid if e and e in texto]
    if codigo == 1:
        evid = []
    return {"codigo": codigo, "prob_si": prob,
            "explicacion": data.get("explicacion", ""), "evidencias": evid}, traza


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Clasificador de confianza (Exp 22)")
    ap.add_argument("variable", choices=list(tools.SKILLS_VARIABLE))
    ap.add_argument("texto")
    ap.add_argument("--modelo", default="gpt-4o-mini")
    a = ap.parse_args()
    res, tz = clasificar(a.variable, a.texto, a.modelo)
    print(json.dumps({"resultado": res, "traza": tz}, ensure_ascii=False, indent=2))
