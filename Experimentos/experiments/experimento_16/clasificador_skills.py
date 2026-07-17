"""
Clasificador de variables con Claude Skills.

Estructura de skills (Experimento 16):
  skills/orquestador/SKILL.md         → orquestador
  skills/<variable>/SKILL.md          → metodología por variable (generadas desde variables.json)
  generar_skills.py                   → regenera las skills/<var>/SKILL.md
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import ValidationError

EXPERIMENTO_DIR = Path(__file__).resolve().parent
EXPERIMENTOS_DIR = EXPERIMENTO_DIR.parent.parent
SKILLS_DIR = EXPERIMENTO_DIR / "skills"
ORQUESTADOR_SKILL_PATH = SKILLS_DIR / "orquestador" / "SKILL.md"

if str(EXPERIMENTOS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTOS_DIR))
if str(EXPERIMENTO_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTO_DIR))

from skill_content import cargar_skill_variable  # noqa: E402
from utils import (  # noqa: E402
    BloqueAnalisisBinario,
    BloqueAnalisisLenguajeSexista,
    _aplanar_ejemplos_negativos,
    _generar_lista_opciones,
    cargar_variables_desde_json,
    obtener_config_variable,
    parsear_respuesta_modelo,
)

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 8192

# Códigos de variable en variables.json → (nombre, tipo de resultado)
VARIABLES_EXP16: dict[str, tuple[str, type]] = {
    "25": ("lenguaje_sexista", BloqueAnalisisLenguajeSexista),
    "26": ("masc_generico", BloqueAnalisisBinario),
    "30": ("sexismo_discurso", BloqueAnalisisBinario),
    "33": ("asimetria_mujer_hombre", BloqueAnalisisBinario),
    "35": ("denominacion_sexualizada", BloqueAnalisisBinario),
}


def cargar_skill_orquestador() -> str:
    if not ORQUESTADOR_SKILL_PATH.is_file():
        raise FileNotFoundError(f"No existe {ORQUESTADOR_SKILL_PATH}")
    return ORQUESTADOR_SKILL_PATH.read_text(encoding="utf-8")


def cargar_system_prompt() -> str:
    """
    System prompt = orquestador + skills/<variable>/SKILL.md (una por variable).
    """
    bloques = [cargar_skill_orquestador()]
    for _, (nombre, _) in VARIABLES_EXP16.items():
        bloques.append(cargar_skill_variable(nombre))
    return "\n\n".join(bloques)


def _config_a_entrada_codebook(config: dict[str, Any]) -> dict[str, Any]:
    valores = config["valores_posibles"]
    entrada: dict[str, Any] = {
        "nombre": config["nombre"],
        "codigo_variable": config.get("codigo", ""),
        "skill_metodologia": config["nombre"],
        "codigos": {str(i + 1): val for i, val in enumerate(valores)},
        "lista_codigos": _generar_lista_opciones(valores),
    }
    if "ejemplos_positivos" in config:
        entrada["ejemplos_positivos"] = config["ejemplos_positivos"]
    elif "ejemplos" in config:
        entrada["ejemplos_positivos"] = config["ejemplos"]
    if config.get("ejemplos_negativos"):
        entrada["ejemplos_negativos"] = _aplanar_ejemplos_negativos(
            config["ejemplos_negativos"]
        )
    if config.get("frontera_con_otras"):
        entrada["frontera_con_otras"] = config["frontera_con_otras"]
    if config.get("caso_limite_documentado"):
        entrada["caso_limite_documentado"] = config["caso_limite_documentado"]
    return entrada


def construir_codebook(ruta_json: str) -> dict[str, Any]:
    vars_data = cargar_variables_desde_json(ruta_json)
    variables = [
        _config_a_entrada_codebook(obtener_config_variable(vars_data, codigo))
        for codigo in VARIABLES_EXP16
    ]
    return {
        "documento": {
            "idioma": "es",
            "experimento": "16",
            "fuente_metodologia": "skills/<variable>/SKILL.md (generadas desde variables.json)",
        },
        "variables": variables,
        "reglas_globales": [
            "La metodología operativa está en las skills anexas al system prompt; ejecútala antes de codificar.",
            "Procesa las variables en el orden declarado.",
            "codigo=1 implica evidencias=[].",
            "Las evidencias deben ser spans literales del texto original.",
            "La explicacion debe citar los pasos de metodologia aplicados.",
        ],
    }


def construir_prompt_usuario(texto: str, codebook: dict[str, Any]) -> str:
    codebook_yaml = yaml.dump(
        codebook,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    nombres = [v["nombre"] for v in codebook["variables"]]
    return (
        "Experimento 16: clasifica el artículo completo.\n"
        "Antes de responder, aplica la metodología de cada skill (system prompt) "
        "en el orden del codebook.\n\n"
        "=== CODEBOOK (códigos y referencia) ===\n"
        f"{codebook_yaml}\n"
        "=== TEXTO ===\n"
        f"{texto}\n"
        "=== FIN ===\n\n"
        "Devuelve únicamente el JSON bajo `variables` con "
        f"({', '.join(nombres)}). "
        "Cada explicacion debe citar los pasos de metodologia ejecutados. "
        "Sin texto adicional ni code fences."
    )


def _limpiar_json_modelo(contenido: str) -> str:
    contenido = contenido.strip()
    contenido = re.sub(r"^```(?:json)?\s*", "", contenido)
    contenido = re.sub(r"\s*```$", "", contenido)
    return contenido


def _verificar_evidencias(
    bloque: dict[str, Any], texto_original: str
) -> dict[str, Any]:
    evidencias = bloque.get("evidencias") or []
    if not isinstance(evidencias, list):
        evidencias = [str(evidencias)] if evidencias else []
    bloque["evidencias"] = [e for e in evidencias if e and e in texto_original]
    if bloque.get("codigo") == 1:
        bloque["evidencias"] = []
    return bloque


def _llamar_modelo(texto: str, codebook: dict[str, Any]) -> dict[str, Any]:
    from anthropic import Anthropic

    cliente = Anthropic()
    respuesta = cliente.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=0.1,
        system=cargar_system_prompt(),
        messages=[{"role": "user", "content": construir_prompt_usuario(texto, codebook)}],
    )
    contenido = "".join(
        bloque.text for bloque in respuesta.content if bloque.type == "text"
    )
    contenido = _limpiar_json_modelo(contenido)
    try:
        data = json.loads(contenido)
    except json.JSONDecodeError as exc:
        data = parsear_respuesta_modelo(contenido)
        if not isinstance(data, dict):
            raise ValueError(
                f"El modelo no devolvió JSON parseable. Primeros 400 chars:\n{contenido[:400]}"
            ) from exc
    if "variables" in data and isinstance(data["variables"], dict):
        return data["variables"]
    return data


def _bloque_por_defecto(
    modelo_cls: type, explicacion: str
) -> Union[BloqueAnalisisLenguajeSexista, BloqueAnalisisBinario]:
    return modelo_cls(codigo=1, explicacion=explicacion, evidencias=[])


def clasificar_articulo(
    texto_articulo: str,
    ruta_json: str,
) -> dict[str, Union[BloqueAnalisisLenguajeSexista, BloqueAnalisisBinario]]:
    """
    Clasifica las 5 variables del experimento 16 en una sola llamada con skills.
    Devuelve dict con claves 'lenguaje_sexista', 'masc_generico', etc.
    """
    texto = texto_articulo.replace('"', "'")
    codebook = construir_codebook(ruta_json)

    try:
        bruto = _llamar_modelo(texto, codebook)
    except Exception as e:
        print(f"❌ Error en llamada con skills: {e}")
        return {
            nombre: _bloque_por_defecto(modelo_cls, f"Error API/skills: {e}")
            for _, (nombre, modelo_cls) in VARIABLES_EXP16.items()
        }

    resultados: dict[str, Union[BloqueAnalisisLenguajeSexista, BloqueAnalisisBinario]] = {}
    for _, (nombre, modelo_cls) in VARIABLES_EXP16.items():
        bloque_raw = bruto.get(nombre)
        if not isinstance(bloque_raw, dict):
            resultados[nombre] = _bloque_por_defecto(
                modelo_cls, f"Variable '{nombre}' ausente en la respuesta del modelo."
            )
            continue
        bloque_raw = _verificar_evidencias(bloque_raw, texto_articulo)
        try:
            resultados[nombre] = modelo_cls(**bloque_raw)
        except ValidationError as e:
            print(f"⚠️ Validación fallida para {nombre}: {e}")
            resultados[nombre] = _bloque_por_defecto(
                modelo_cls, f"Respuesta inválida: {e}"
            )
    return resultados
