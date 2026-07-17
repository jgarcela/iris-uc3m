"""
anotador.py — Pipeline de anotación periodística contra un codebook.

Patrón: pipeline secuencial con verificación.
  Fase 1: extracción (un solo agente con la skill cargada)
  Fase 2: verificación de spans literales (determinista, no LLM)
  Fase 3: validación de schema y consistencia con codebook (determinista)

Uso:
    export ANTHROPIC_API_KEY=sk-...
    python anotador.py \
        --texto examples/articulo.txt \
        --codebook codebooks/ejemplo_genero_politica.yaml \
        --salida examples/articulo.anotado.json

Dependencias:
    pip install anthropic pyyaml jsonschema
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from anthropic import Anthropic
from jsonschema import Draft7Validator

ROOT = Path(__file__).resolve().parent.parent
SKILL_PATH = ROOT / "skill" / "SKILL.md"
SCHEMA_PATH = ROOT / "schemas" / "anotacion.schema.json"

MODEL = "claude-haiku-4-5-20251001" # "claude-sonnet-4-6" # "claude-opus-4-7"
MAX_TOKENS = 8000


@dataclass
class ResultadoAnotacion:
    """Resultado consolidado tras las tres fases."""

    json_anotaciones: dict[str, Any]
    spans_descartados: list[dict[str, Any]] = field(default_factory=list)
    etiquetas_invalidas: list[dict[str, Any]] = field(default_factory=list)
    errores_schema: list[str] = field(default_factory=list)

    @property
    def es_valido(self) -> bool:
        return not (self.errores_schema or self.etiquetas_invalidas)

    @property
    def num_anotaciones(self) -> int:
        return len(self.json_anotaciones.get("anotaciones", []))


def cargar_skill() -> str:
    """Carga el contenido del SKILL.md como instrucción del sistema."""
    return SKILL_PATH.read_text(encoding="utf-8")


def cargar_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def cargar_codebook(ruta: Path) -> dict[str, Any]:
    return yaml.safe_load(ruta.read_text(encoding="utf-8"))


def construir_prompt_usuario(texto: str, codebook: dict[str, Any]) -> str:
    """Construye el mensaje de usuario con texto y codebook serializados."""
    codebook_yaml = yaml.dump(
        codebook,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return (
        "Anota el siguiente texto periodístico contra el codebook.\n\n"
        "=== CODEBOOK ===\n"
        f"{codebook_yaml}\n"
        "=== TEXTO ===\n"
        f"{texto}\n"
        "=== FIN ===\n\n"
        "Devuelve únicamente el JSON descrito en la skill, sin texto adicional ni code fences."
    )


def fase_1_extraer(texto: str, codebook: dict[str, Any]) -> dict[str, Any]:
    """Llama al modelo cargando la skill como system prompt."""
    cliente = Anthropic()
    respuesta = cliente.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=cargar_skill(),
        messages=[
            {"role": "user", "content": construir_prompt_usuario(texto, codebook)},
        ],
    )

    contenido = "".join(
        bloque.text for bloque in respuesta.content if bloque.type == "text"
    ).strip()

    contenido = re.sub(r"^```(?:json)?\s*", "", contenido)
    contenido = re.sub(r"\s*```$", "", contenido)

    try:
        return json.loads(contenido)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"El modelo no devolvió JSON parseable. Primeros 400 chars:\n{contenido[:400]}"
        ) from exc


def fase_2_verificar_spans(
    resultado: dict[str, Any], texto_original: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Descarta toda anotación cuyo span no aparezca literalmente en el texto."""
    descartadas: list[dict[str, Any]] = []
    anotaciones_validas: list[dict[str, Any]] = []

    for anotacion in resultado.get("anotaciones", []):
        span = anotacion.get("texto", "")
        if not span or span not in texto_original:
            descartadas.append(
                {
                    "id": anotacion.get("id"),
                    "span_solicitado": span,
                    "razon": "span no encontrado literalmente en el texto original",
                }
            )
            continue
        anotaciones_validas.append(anotacion)

    resultado["anotaciones"] = anotaciones_validas
    return resultado, descartadas


def fase_3_validar_codebook(
    resultado: dict[str, Any], codebook: dict[str, Any]
) -> list[dict[str, Any]]:
    """Verifica que cada (variable, etiqueta) exista en el codebook."""
    catalogo: dict[str, set[str]] = {
        v["nombre"]: set(v["etiquetas"]) for v in codebook.get("variables", [])
    }
    invalidas: list[dict[str, Any]] = []

    for anotacion in resultado.get("anotaciones", []):
        var = anotacion.get("variable")
        etq = anotacion.get("etiqueta")
        if var not in catalogo:
            invalidas.append({"id": anotacion.get("id"), "razon": f"variable '{var}' no existe"})
        elif etq not in catalogo[var]:
            invalidas.append(
                {
                    "id": anotacion.get("id"),
                    "razon": f"etiqueta '{etq}' no permitida para variable '{var}'",
                    "etiquetas_validas": sorted(catalogo[var]),
                }
            )

    return invalidas


def fase_4_validar_schema(resultado: dict[str, Any]) -> list[str]:
    """Valida estructuralmente con JSON Schema."""
    validador = Draft7Validator(cargar_schema())
    return [
        f"{'/'.join(str(p) for p in err.absolute_path)}: {err.message}"
        for err in validador.iter_errors(resultado)
    ]


def recalcular_resumen(resultado: dict[str, Any]) -> dict[str, Any]:
    """Recalcula el bloque resumen tras posibles descartes."""
    anotaciones = resultado.get("anotaciones", [])
    por_variable: dict[str, dict[str, int]] = {}
    for a in anotaciones:
        v, e = a["variable"], a["etiqueta"]
        por_variable.setdefault(v, {}).setdefault(e, 0)
        por_variable[v][e] += 1

    resultado.setdefault("resumen", {})
    resultado["resumen"]["total_anotaciones"] = len(anotaciones)
    resultado["resumen"]["por_variable"] = por_variable
    return resultado


def anotar(texto: str, codebook: dict[str, Any]) -> ResultadoAnotacion:
    """Ejecuta el pipeline completo y devuelve un ResultadoAnotacion."""
    bruto = fase_1_extraer(texto, codebook)
    bruto, descartadas = fase_2_verificar_spans(bruto, texto)
    invalidas = fase_3_validar_codebook(bruto, codebook)

    if invalidas:
        ids_invalidas = {i["id"] for i in invalidas}
        bruto["anotaciones"] = [
            a for a in bruto["anotaciones"] if a["id"] not in ids_invalidas
        ]

    bruto = recalcular_resumen(bruto)
    errores_schema = fase_4_validar_schema(bruto)

    return ResultadoAnotacion(
        json_anotaciones=bruto,
        spans_descartados=descartadas,
        etiquetas_invalidas=invalidas,
        errores_schema=errores_schema,
    )


def renderizar_html_marcado(texto: str, resultado: dict[str, Any]) -> str:
    """Genera HTML del texto con highlights por anotación. Ordena por inicio."""
    coloreado = {
        "atribucion_genero": "#EEEDFE",
        "rol_tematico": "#E1F5EE",
        "descriptor_sexista": "#FAECE7",
        "frame_dominante": "#FBEAF0",
        "cita_directa": "#F1EFE8",
    }

    eventos = []
    for a in resultado.get("anotaciones", []):
        inicio = texto.find(a["texto"])
        if inicio < 0:
            continue
        eventos.append((inicio, inicio + len(a["texto"]), a))
    eventos.sort(key=lambda e: (e[0], -e[1]))

    cursor = 0
    fragmentos: list[str] = []
    for inicio, fin, a in eventos:
        if inicio < cursor:
            continue
        fragmentos.append(texto[cursor:inicio].replace("\n", "<br>"))
        color = coloreado.get(a["variable"], "#F1EFE8")
        fragmentos.append(
            f'<mark style="background:{color};padding:2px 4px;border-radius:3px;" '
            f'title="{a["variable"]} → {a["etiqueta"]} ({a["confianza"]:.2f})">'
            f'{texto[inicio:fin]}'
            f'<sub style="font-size:10px;opacity:0.7;">{a["variable"][:3]}</sub>'
            f'</mark>'
        )
        cursor = fin
    fragmentos.append(texto[cursor:].replace("\n", "<br>"))
    return "".join(fragmentos)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texto", required=True, type=Path)
    parser.add_argument("--codebook", required=True, type=Path)
    parser.add_argument("--salida", required=True, type=Path)
    parser.add_argument("--html", type=Path, help="Opcional: ruta para HTML marcado")
    args = parser.parse_args()

    texto = args.texto.read_text(encoding="utf-8").strip()
    codebook = cargar_codebook(args.codebook)

    resultado = anotar(texto, codebook)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    args.salida.write_text(
        json.dumps(resultado.json_anotaciones, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Anotaciones válidas: {resultado.num_anotaciones}", file=sys.stderr)
    if resultado.spans_descartados:
        print(
            f"Spans descartados (no literales): {len(resultado.spans_descartados)}",
            file=sys.stderr,
        )
    if resultado.etiquetas_invalidas:
        print(
            f"Etiquetas inválidas: {len(resultado.etiquetas_invalidas)}",
            file=sys.stderr,
        )
        for inv in resultado.etiquetas_invalidas:
            print(f"  · {inv}", file=sys.stderr)
    if resultado.errores_schema:
        print("Errores de schema:", file=sys.stderr)
        for err in resultado.errores_schema:
            print(f"  · {err}", file=sys.stderr)

    if args.html:
        args.html.write_text(
            renderizar_html_marcado(texto, resultado.json_anotaciones),
            encoding="utf-8",
        )
        print(f"HTML escrito en {args.html}", file=sys.stderr)

    return 0 if resultado.es_valido else 1


if __name__ == "__main__":
    sys.exit(main())
