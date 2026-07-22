"""
Registro de skills y tools del Experimento 21 (Agent Skills).

Progressive disclosure: el agente sólo ve `name` + `description` (metadatos) vía
`list_skills`; el cuerpo del SKILL.md se carga bajo demanda con `read_skill`.
No depende de LangChain ni de un proveedor concreto: las tools se exponen como
acciones de texto que agente.py enruta (ver protocolo en agente.py).
"""

from __future__ import annotations

import re
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# Skills de variable (una por variable objetivo) — el agente carga la suya.
SKILLS_VARIABLE = {
    "lenguaje_sexista": "25",
    "masc_generico": "26",
    "sexismo_discurso": "30",
    "asimetria_mujer_hombre": "33",
    "denominacion_sexualizada": "35",
}

# Skills auxiliares compartidas — cualquier agente las carga bajo demanda.
SKILLS_AUXILIARES = (
    "guia_regla_inversion",
    "guia_lenguaje_inclusivo",
    "verificar_evidencias",
)

# Resúmenes de guías reales (auto-generados por generar_resumenes_guias.py).
# Descubribles vía list_skills(); complementan la tool CONSULTAR_GUIA (texto literal).
SKILLS_RESUMEN_GUIAS = (
    "guia_resumen_csd_deporte",
    "guia_resumen_inclusivo_justicia",
    "guia_resumen_no_sexista",
    "guia_resumen_uso_igualitario",
    "guia_resumen_igualdad",
)


def _leer_frontmatter(path: Path) -> dict:
    """Extrae name/description del frontmatter YAML simple de un SKILL.md."""
    texto = path.read_text(encoding="utf-8")
    m = re.match(r"^---\s*\n(.*?)\n---", texto, re.DOTALL)
    campos: dict[str, str] = {}
    if m:
        for linea in m.group(1).splitlines():
            if ":" in linea:
                k, _, v = linea.partition(":")
                campos[k.strip()] = v.strip()
    campos.setdefault("name", path.parent.name)
    campos.setdefault("description", "")
    return campos


def _skill_path(skill_id: str) -> Path:
    return SKILLS_DIR / skill_id / "SKILL.md"


def skills_disponibles() -> list[str]:
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(
        p.name for p in SKILLS_DIR.iterdir()
        if p.is_dir() and (p / "SKILL.md").is_file()
    )


def list_skills(ids: list[str] | None = None) -> str:
    """
    Devuelve metadatos (id + description) de las skills visibles para el agente.
    `ids`: subconjunto a mostrar (p.ej. la skill de la variable + auxiliares).
    """
    disponibles = skills_disponibles()
    if ids is not None:
        disponibles = [s for s in disponibles if s in ids]
    if not disponibles:
        return "(no hay skills disponibles)"
    lineas = ["Skills disponibles (usa LEER_SKILL: <id>):"]
    for sid in disponibles:
        desc = _leer_frontmatter(_skill_path(sid)).get("description", "")
        lineas.append(f"- {sid}: {desc}")
    return "\n".join(lineas)


def read_skill(skill_id: str, permitidas: list[str] | None = None) -> str:
    """Devuelve el cuerpo completo del SKILL.md (sin frontmatter)."""
    if permitidas is not None and skill_id not in permitidas:
        return (
            f"Error: '{skill_id}' no está permitida aquí. "
            f"Permitidas: {', '.join(permitidas)}"
        )
    path = _skill_path(skill_id)
    if not path.is_file():
        return f"Error: no existe la skill '{skill_id}'."
    texto = path.read_text(encoding="utf-8")
    return re.sub(r"^---\s*\n.*?\n---\s*\n", "", texto, count=1, flags=re.DOTALL).strip()


def verificar_evidencias(evidencias: list[str], texto: str) -> dict:
    """
    Tool de verificación: separa evidencias que SON literales del texto de las
    que no. `texto` debe ser el MISMO que vio el modelo (no un texto alterado).
    """
    validas = [e for e in evidencias if e and e in texto]
    invalidas = [e for e in evidencias if e and e not in texto]
    return {"validas": validas, "invalidas": invalidas}
