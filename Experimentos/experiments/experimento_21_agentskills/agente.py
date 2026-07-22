"""
Agente especializado por variable (Experimento 21 — Agent Skills).

Un agente = una variable. Bucle de decisión estilo ReAct sobre acciones de texto
(no tool-calling nativo → uniforme en Anthropic/OpenAI/Gemini/Ollama vía utils.py):

  - El modelo ve sólo name+description de sus skills (progressive disclosure).
  - Emite acciones de texto:
        LEER_SKILL: <id>        → le devolvemos el cuerpo del SKILL.md
        VERIFICAR: ["..","..."] → verificamos que son literales del texto
        FINAL: {json}           → veredicto de la variable
  - Si el modelo emite FINAL sin usar ninguna skill, se registra "colapso a B0".

Métrica de comportamiento devuelta en `traza`: nº de tools, skills cargadas,
si colapsó a B0. Sirve para el Cap. 5 (uso real de tools por proveedor).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

EXP_DIR = Path(__file__).resolve().parent
EXPERIMENTOS_DIR = EXP_DIR.parent.parent
for p in (str(EXPERIMENTOS_DIR), str(EXP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils import IRIS_CACHE_BREAK, consultar_ollama, get_consumo_llamada  # noqa: E402
import tools  # noqa: E402
import guias  # noqa: E402
import costes  # noqa: E402

MAX_ITERS = 8


SYSTEM_TMPL = """Eres una experta en análisis de género en medios y clasificas ÚNICAMENTE la variable «{variable}».

La metodología NO está en este mensaje: está en disco, en skills que debes cargar.

{catalogo}

Trabaja por acciones. En cada turno responde con UNA sola acción, exactamente en una de estas formas:

  LEER_SKILL: <id>
      Carga el cuerpo de esa skill. Carga SIEMPRE primero tu skill de variable ({variable}).
      Carga una skill auxiliar sólo si la metodología lo pide o si dudas.

  CONSULTAR_GUIA: <consulta en lenguaje natural>
      Recupera pasajes LITERALES de las guías expertas de lenguaje (Sainz de Baranda,
      CSD, guías de lenguaje inclusivo…) para fundamentar un caso dudoso. Cita la fuente.

  VERIFICAR: ["evidencia 1", "evidencia 2"]
      Comprueba que esos fragmentos son literales del texto antes de cerrar.

  FINAL: {{"codigo": <n>, "explicacion": "...", "evidencias": [...]}}
      Veredicto de «{variable}». `codigo` según la skill; evidencias literales
      del texto (o [] si codigo=1). La explicacion cita los pasos aplicados.

Reglas: no inventes el contenido de una skill sin haberla leído. No mezcles otras variables."""


# Ablación de catálogo: si es False, las skills-resumen de guías no se listan ni se
# pueden cargar (el agente sigue teniendo su skill de variable + auxiliares + CONSULTAR_GUIA).
INCLUIR_RESUMENES_GUIAS = True

# Ablación de prompt caching: si es True se inserta IRIS_CACHE_BREAK, que además de
# habilitar la caché reestructura el prompt (prefijo→system, sufijo→user). Ese cambio
# de estructura altera el comportamiento del agente: ver DIARIO/Cap.5.
USAR_PROMPT_CACHE = True


def _cache_break() -> str:
    return IRIS_CACHE_BREAK if USAR_PROMPT_CACHE else "\n\n"


SYSTEM_BASELINE = """Eres una experta en análisis de género en medios y clasificas ÚNICAMENTE la variable «{variable}».

A continuación tienes la metodología completa que debes aplicar.

--- METODOLOGÍA: {variable} ---
{metodologia}
--- FIN METODOLOGÍA ---

Aplica esa metodología al texto y responde con UNA sola línea, exactamente en esta forma:

  FINAL: {{"codigo": <n>, "explicacion": "...cita los pasos aplicados...", "evidencias": [literales del texto, o [] si codigo=1]}}

No añadas nada más. No mezcles otras variables."""


def clasificar_variable_baseline(
    variable: str,
    texto: str,
    modelo: str = "claude-haiku-4-5-20251001",
    temperature: float = 0.1,
    verbose: bool = False,
) -> tuple[dict, dict]:
    """
    Nivel B0: misma tarea y mismas 5 llamadas, pero la metodología va INYECTADA en el
    prompt (sin progressive disclosure, sin tools). Única diferencia frente a B1.
    """
    system = SYSTEM_BASELINE.format(
        variable=variable, metodologia=tools.read_skill(variable))
    traza = {"skills_cargadas": [], "guias_consultadas": [], "n_tools": 0,
             "verifico": False, "colapso_b0": True, "iters": 1, "error": None,
             "prompt_tokens": 0, "completion_tokens": 0, "cache_read_tokens": 0,
             "cache_creation_tokens": 0, "n_llamadas": 0, "coste_usd": None}

    prompt = (system + "\n\n=== TEXTO A CLASIFICAR ===\n" + texto
              + "\n=== FIN TEXTO ===\n\nTu respuesta:")
    salida = consultar_ollama(prompt, modelo=modelo, temperature=temperature)
    c = get_consumo_llamada()
    for k in ("prompt_tokens", "completion_tokens", "cache_read_tokens",
              "cache_creation_tokens"):
        traza[k] += c.get(k, 0)
    traza["n_llamadas"] = 1
    if verbose:
        print(f"[B0 {variable}] {salida[:200]}")

    accion, arg = _parse_accion(salida)
    traza["coste_usd"] = costes.calcular_coste(
        modelo, traza["prompt_tokens"], traza["completion_tokens"],
        traza["cache_read_tokens"], traza["cache_creation_tokens"])
    if accion == "FINAL":
        return _sanear_resultado(arg, texto), traza
    traza["error"] = "sin_final"
    return {"codigo": 1, "explicacion": "El modelo no devolvió FINAL.",
            "evidencias": []}, traza


def _catalogo_ids(variable: str) -> list[str]:
    ids = [variable] + list(tools.SKILLS_AUXILIARES)
    if INCLUIR_RESUMENES_GUIAS:
        ids += list(tools.SKILLS_RESUMEN_GUIAS)
    return ids


def _construir_catalogo(variable: str) -> str:
    return tools.list_skills(_catalogo_ids(variable))


def _permitidas(variable: str) -> list[str]:
    return _catalogo_ids(variable)


def _parse_accion(salida: str) -> tuple[str, Any]:
    s = salida.strip()
    m = re.search(r"FINAL:\s*(\{.*\})", s, re.DOTALL)
    if m:
        try:
            return "FINAL", json.loads(m.group(1))
        except json.JSONDecodeError:
            return "FINAL_MALO", m.group(1)
    m = re.search(r"LEER_SKILL:\s*([a-z_0-9]+)", s)
    if m:
        return "LEER_SKILL", m.group(1)
    m = re.search(r"CONSULTAR_GUIA:\s*(.+)", s)
    if m:
        return "CONSULTAR_GUIA", m.group(1).strip().splitlines()[0]
    m = re.search(r"VERIFICAR:\s*(\[.*\])", s, re.DOTALL)
    if m:
        try:
            return "VERIFICAR", json.loads(m.group(1))
        except json.JSONDecodeError:
            return "VERIFICAR", []
    return "DESCONOCIDO", s


def clasificar_variable(
    variable: str,
    texto: str,
    modelo: str = "claude-haiku-4-5-20251001",
    temperature: float = 0.1,
    verbose: bool = False,
) -> tuple[dict, dict]:
    """
    Ejecuta el agente especializado en `variable` sobre `texto`.
    Devuelve (resultado, traza). traza = métricas de comportamiento del agente.
    """
    system = SYSTEM_TMPL.format(variable=variable, catalogo=_construir_catalogo(variable))
    permitidas = _permitidas(variable)
    historial = [f"=== TEXTO A CLASIFICAR ===\n{texto}\n=== FIN TEXTO ==="]

    traza = {"skills_cargadas": [], "guias_consultadas": [], "n_tools": 0,
             "verifico": False, "colapso_b0": False, "iters": 0, "error": None,
             "prompt_tokens": 0, "completion_tokens": 0, "cache_read_tokens": 0,
             "cache_creation_tokens": 0, "n_llamadas": 0, "coste_usd": None}

    def _llamar(prompt_txt: str) -> str:
        """Llama al modelo y acumula el consumo de tokens en la traza."""
        salida = consultar_ollama(prompt_txt, modelo=modelo, temperature=temperature)
        c = get_consumo_llamada()
        traza["prompt_tokens"] += c.get("prompt_tokens", 0)
        traza["completion_tokens"] += c.get("completion_tokens", 0)
        traza["cache_read_tokens"] += c.get("cache_read_tokens", 0)
        traza["cache_creation_tokens"] += c.get("cache_creation_tokens", 0)
        traza["n_llamadas"] += 1
        return salida

    def _cerrar(resultado_raw):
        traza["coste_usd"] = costes.calcular_coste(
            modelo, traza["prompt_tokens"], traza["completion_tokens"],
            traza["cache_read_tokens"], traza["cache_creation_tokens"])
        return _sanear_resultado(resultado_raw, texto), traza

    for i in range(MAX_ITERS):
        traza["iters"] = i + 1
        # Últimos 2 turnos: obliga a cerrar para no agotar el presupuesto sin veredicto.
        cierre = ""
        if i >= MAX_ITERS - 2:
            cierre = ("\n\n[Sistema] Te quedan pocos turnos. Con lo que ya sabes, responde "
                      "AHORA únicamente con `FINAL: {...}`. No cargues más skills.")
        # Prompt caching: el corte va al final del historial acumulado. Cada iteración
        # reutiliza como prefijo cacheado todo lo anterior (system + texto + SKILL.md
        # ya cargados), que es donde está el grueso de tokens.
        prompt = (system + "\n\n" + "\n\n".join(historial)
                  + _cache_break() + cierre + "\n\nTu acción:")
        salida = _llamar(prompt)
        if verbose:
            print(f"[{variable} iter {i}] {salida[:200]}")
        accion, arg = _parse_accion(salida)

        if accion == "LEER_SKILL":
            traza["n_tools"] += 1
            cuerpo = tools.read_skill(arg, permitidas=permitidas)
            traza["skills_cargadas"].append(arg)
            historial.append(f"[Acción] LEER_SKILL: {arg}")
            historial.append(f"[Resultado skill {arg}]\n{cuerpo}")
        elif accion == "CONSULTAR_GUIA":
            traza["n_tools"] += 1
            traza["guias_consultadas"].append(arg)
            pasajes = guias.consultar_guia(arg, k=2)
            historial.append(f"[Acción] CONSULTAR_GUIA: {arg}")
            historial.append(f"[Pasajes de guías]\n{pasajes}")
        elif accion == "VERIFICAR":
            traza["n_tools"] += 1
            traza["verifico"] = True
            res = tools.verificar_evidencias(arg, texto)
            historial.append(f"[Acción] VERIFICAR")
            historial.append(f"[Resultado] válidas={res['validas']} inválidas={res['invalidas']}")
        elif accion == "FINAL":
            if traza["n_tools"] == 0:  # cerró sin usar ninguna tool → equivalente a B0
                traza["colapso_b0"] = True
            return _cerrar(arg)
        else:  # FINAL_MALO / DESCONOCIDO
            historial.append(
                "[Sistema] Acción no reconocida. Responde con LEER_SKILL:, VERIFICAR: o FINAL:."
            )

    # Presupuesto agotado sin FINAL: un último intento que SOLO pide el veredicto.
    prompt = (system + "\n\n" + "\n\n".join(historial) + _cache_break() +
              "\n\n[Sistema] Cierra ya. Responde EXCLUSIVAMENTE con "
              "`FINAL: {\"codigo\": <n>, \"explicacion\": \"...\", \"evidencias\": [...]}`.")
    salida = _llamar(prompt)
    accion, arg = _parse_accion(salida)
    if accion == "FINAL":
        return _cerrar(arg)
    traza["error"] = "sin_final"
    return _cerrar({"codigo": 1,
                    "explicacion": "El agente no emitió FINAL (presupuesto agotado).",
                    "evidencias": []})


def _sanear_resultado(res: Any, texto: str) -> dict:
    if not isinstance(res, dict):
        return {"codigo": 1, "explicacion": "Resultado no es objeto.", "evidencias": []}
    codigo = res.get("codigo", 1)
    evidencias = res.get("evidencias") or []
    if not isinstance(evidencias, list):
        evidencias = [str(evidencias)]
    # Evidencias contra el MISMO texto que vio el modelo (corrige bug exp16).
    evidencias = [e for e in evidencias if e and e in texto]
    if codigo == 1:
        evidencias = []
    return {"codigo": codigo, "explicacion": res.get("explicacion", ""), "evidencias": evidencias}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Agente especializado de una variable")
    ap.add_argument("variable", choices=list(tools.SKILLS_VARIABLE))
    ap.add_argument("texto")
    ap.add_argument("--modelo", default="claude-haiku-4-5-20251001")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    resultado, traza = clasificar_variable(a.variable, a.texto, modelo=a.modelo, verbose=a.verbose)
    print(json.dumps({"resultado": resultado, "traza": traza}, ensure_ascii=False, indent=2))
