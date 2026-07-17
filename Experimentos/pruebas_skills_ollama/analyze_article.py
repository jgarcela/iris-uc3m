"""
Pipeline de análisis de artículo (5 skills lógicas, una sola inferencia, sin RAG).

Lee variables (`vars_lenguaje.md`), guías `.md` del manifiesto (o `--guides`) y
los SKILL del pipeline; llama una vez a Ollama vía LangChain y devuelve JSON unificado.

Ejemplo::

    cd Experimentos/pruebas_skills
    source .venv/bin/activate
    pip install -r requirements.txt
    python analyze_article.py --article muestra.txt -o salida.json
    python analyze_article.py --article muestra.txt --guides CSD_GUIA_PER_DEPOR.md --max-total-guides 80000

Variables de entorno: ``OLLAMA_MODEL`` si no pasas ``--model``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent
_DEFAULT_MANIFEST = _ROOT / "methodology_manifest.json"
_PIPELINE_SKILLS = [
    "pipeline_01_carga_contexto",
    "pipeline_02_deteccion_etiquetado",
    "pipeline_03_highlighting",
    "pipeline_04_explicacion",
    "pipeline_05_alternativas",
]

class ArticleMeta(BaseModel):
    """Metadatos del artículo analizado (bloque bajo ## Texto del artículo)."""

    titular: str | None = None
    longitud_caracteres: int = 0
    notas: str | None = None


class LabelEntry(BaseModel):
    valor_etiqueta: str = Field(description="Etiqueta según valores posibles del catálogo")
    notas: str = Field(default="", description="Justificación breve")


class Finding(BaseModel):
    finding_id: int
    variable_codigo: str = Field(description="Código de variable 25-39 como string")
    start: int = Field(description="Índice inicio carácter Python en el artículo")
    end: int = Field(description="Índice fin exclusivo o inclusivo según excerpt; usar exclusivo típico slice")
    excerpt: str
    explicacion: str
    guia_referencia: str = Field(description="Nombre de archivo de guía y criterio")
    severidad: Literal["baja", "media", "alta"]


class Alternative(BaseModel):
    finding_id: int
    texto_original: str
    texto_propuesto: str
    notas: str | None = None


class UnifiedOutput(BaseModel):
    """Contrato JSON unificado del pipeline (diagrama)."""

    article_meta: ArticleMeta
    labels: dict[str, LabelEntry] = Field(
        default_factory=dict,
        description="Claves = código variable string p.ej. 25",
    )
    findings: list[Finding] = Field(default_factory=list)
    alternatives: list[Alternative] = Field(default_factory=list)
    annotated_summary: str | None = None


_JSON_CONTRACT_HINT = """
La salida se valida contra un esquema fijo (article_meta, labels, findings, alternatives, annotated_summary).
Analiza solo el bloque "## Texto del artículo"; el resto son guías y catálogo de referencia.
Offsets en caracteres Python (str) sobre ese texto.
"""


def limpiar_json(texto: str | dict) -> dict:
    """Extrae JSON de la respuesta del modelo (misma idea que pruebas_embeddings/rag/agents.py)."""
    if isinstance(texto, dict):
        return texto
    if not isinstance(texto, str):
        texto = str(texto)
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texto, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    start = texto.find("{")
    end = texto.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(texto[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"raw": texto, "error": "No se pudo parsear el JSON"}


def _safe_guide_path(methodology_dir: Path, name: str) -> Path:
    """Evita path traversal; name es relativo (solo nombre o subruta bajo methodology)."""
    methodology_dir = methodology_dir.resolve()
    candidate = (methodology_dir / name).resolve()
    try:
        candidate.relative_to(methodology_dir)
    except ValueError as e:
        raise ValueError(f"Ruta de guía no permitida: {name}") from e
    if not candidate.is_file():
        raise FileNotFoundError(f"No existe el fichero de guía: {candidate}")
    return candidate


def load_manifest(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    guides = data.get("guides")
    if not isinstance(guides, list):
        raise ValueError("El manifiesto debe contener la clave 'guides' (lista).")
    return [str(x) for x in guides]


def load_guides_block(
    methodology_dir: Path,
    guide_names: list[str],
    max_chars_per_guide: int,
    max_total_guides: int,
) -> str:
    parts: list[str] = []
    budget = max_total_guides
    omitted: list[str] = []

    for name in guide_names:
        path = _safe_guide_path(methodology_dir, name)
        raw = path.read_text(encoding="utf-8")
        if len(raw) > max_chars_per_guide:
            omitted_n = len(raw) - max_chars_per_guide
            raw = (
                raw[:max_chars_per_guide]
                + f"\n\n[TRUNCADO {name}: {omitted_n} caracteres omitidos al final]\n"
            )
        header = f"\n\n=== {name} ===\n\n"
        chunk = header + raw
        if len(chunk) <= budget:
            parts.append(chunk)
            budget -= len(chunk)
        else:
            avail = budget - len(header)
            if avail > 200:
                parts.append(
                    header
                    + raw[:avail]
                    + f"\n\n[TRUNCADO por presupuesto total: quedaban {len(raw) - avail} caracteres en esta guía]\n"
                )
            else:
                omitted.append(name)
            budget = 0
            break

    if omitted:
        parts.append("\n[Guías no cargadas por límite de contexto: " + ", ".join(omitted) + "]\n")

    return "".join(parts).strip()


def load_pipeline_skills_text() -> str:
    blocks: list[str] = []
    for skill_id in _PIPELINE_SKILLS:
        p = _ROOT / "skills" / skill_id / "SKILL.md"
        if not p.is_file():
            raise FileNotFoundError(f"Falta el skill del pipeline: {p}")
        blocks.append(f"\n\n--- SKILL: {skill_id} ---\n\n")
        blocks.append(p.read_text(encoding="utf-8"))
    return "".join(blocks).strip()


def build_messages(
    article_text: str,
    titular: str | None,
    variables_text: str,
    guides_text: str,
    pipeline_text: str,
) -> list:
    system = (
        "Eres un analista de calidad periodística e inclusión de género en medios españoles.\n"
        "Sigues las fases de los SKILL del pipeline en el mensaje de usuario, en orden.\n"
        "No inventes guías ni variables: usa solo el material proporcionado.\n"
        "Si no hay hallazgos, devuelve findings=[] y alternatives=[].\n"
        + _JSON_CONTRACT_HINT
    )
    user_sections = [
        "## Texto del artículo\n",
        article_text.strip(),
        "\n\n## Titular (si aplica)\n",
        (titular or "").strip() or "(no proporcionado)",
        "\n\n## Catálogo de variables (vars_lenguaje)\n",
        variables_text.strip(),
        "\n\n## Guías metodológicas (fragmentos; pueden estar truncadas)\n",
        guides_text or "(sin guías cargadas)",
        "\n\n## Instrucciones del pipeline (Skills 1–5)\n",
        pipeline_text,
        "\n\n## Tarea\n",
        "Ejecuta las fases en orden y produce el JSON unificado descrito en el sistema. "
        "Auto-revisa coherencia de offsets con el texto del artículo antes de responder.",
    ]
    return [
        SystemMessage(content=system),
        HumanMessage(content="".join(user_sections)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline diagrama: una inferencia Ollama, sin RAG.")
    parser.add_argument("--article", required=True, type=Path, help="Ruta al .txt o .md del artículo")
    parser.add_argument("--titular", default="", help="Titular opcional")
    parser.add_argument(
        "--variables",
        type=Path,
        default=_ROOT / "variables" / "vars_lenguaje.md",
        help="Catálogo de variables en markdown",
    )
    parser.add_argument(
        "--methodology-dir",
        type=Path,
        default=_ROOT / "methodology",
        help="Directorio con las guías .md",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="JSON con clave 'guides' (lista de nombres de fichero bajo methodology)",
    )
    parser.add_argument(
        "--guides",
        nargs="*",
        default=None,
        help="Si se indica, sustituye la lista del manifiesto (solo estos .md bajo methodology)",
    )
    parser.add_argument(
        "--max-chars-per-guide",
        type=int,
        default=50_000,
        help="Máximo de caracteres por guía antes de truncar cola",
    )
    parser.add_argument(
        "--max-total-guides",
        type=int,
        default=200_000,
        help="Tope total de caracteres para el bloque de todas las guías",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "gemma4:e4b"),
        help="Modelo Ollama (por defecto env OLLAMA_MODEL o gemma4:e4b)",
    )
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("-o", "--output", type=Path, default=None, help="Escribir JSON parseado aquí")
    args = parser.parse_args()

    article_path = args.article.expanduser().resolve()
    if not article_path.is_file():
        print(f"Error: no existe --article {article_path}", file=sys.stderr)
        sys.exit(1)
    article_text = article_path.read_text(encoding="utf-8")

    variables_path = args.variables.expanduser().resolve()
    if not variables_path.is_file():
        print(f"Error: no existe --variables {variables_path}", file=sys.stderr)
        sys.exit(1)
    variables_text = variables_path.read_text(encoding="utf-8")

    methodology_dir = args.methodology_dir.expanduser().resolve()
    if not methodology_dir.is_dir():
        print(f"Error: no existe --methodology-dir {methodology_dir}", file=sys.stderr)
        sys.exit(1)

    if args.guides is not None and len(args.guides) == 0:
        print("Error: --guides requiere al menos un nombre de fichero", file=sys.stderr)
        sys.exit(1)

    if args.guides is not None:
        guide_names = list(args.guides)
    else:
        manifest_path = args.manifest.expanduser().resolve()
        if not manifest_path.is_file():
            print(f"Error: no existe --manifest {manifest_path}", file=sys.stderr)
            sys.exit(1)
        guide_names = load_manifest(manifest_path)

    try:
        guides_text = load_guides_block(
            methodology_dir,
            guide_names,
            args.max_chars_per_guide,
            args.max_total_guides,
        )
    except (OSError, ValueError) as e:
        print(f"Error cargando guías: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        pipeline_text = load_pipeline_skills_text()
    except OSError as e:
        print(f"Error cargando skills del pipeline: {e}", file=sys.stderr)
        sys.exit(1)

    messages = build_messages(
        article_text,
        args.titular.strip() or None,
        variables_text,
        guides_text,
        pipeline_text,
    )

    llm = ChatOllama(model=args.model, temperature=args.temperature)
    structured = llm.with_structured_output(UnifiedOutput, method="json_schema")
    try:
        out = structured.invoke(messages)
        parsed = out.model_dump() if isinstance(out, UnifiedOutput) else dict(out)
    except Exception as exc:  # noqa: BLE001
        llm_json = ChatOllama(model=args.model, temperature=args.temperature, format="json")
        response = llm_json.invoke(messages)
        raw_out = response.content if hasattr(response, "content") else str(response)
        parsed = limpiar_json(raw_out)
        required = {"article_meta", "labels", "findings", "alternatives"}
        if not isinstance(parsed, dict) or not required.issubset(parsed.keys()):
            parsed = {
                "_schema_warning": (
                    f"Salida estructurada no disponible o inválida ({exc!s}). "
                    "Prueba otro modelo Ollama con soporte de structured outputs o actualiza Ollama."
                ),
                "parsed_partial": parsed,
            }
        else:
            parsed["_note"] = f"Se usó respaldo format=json tras error en json_schema: {exc!s}"

    if isinstance(parsed, dict) and isinstance(parsed.get("article_meta"), dict):
        parsed["article_meta"]["longitud_caracteres"] = len(article_text)

    out_json = json.dumps(parsed, indent=2, ensure_ascii=False)
    print(out_json)
    if args.output:
        out_path = args.output.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_json + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
